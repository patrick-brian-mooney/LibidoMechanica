#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This is a utility script for LibidoMechanica. It's designed to go through the
poems in the source corpus and correct problems in the source texts. There are
a variety of textual transformations applied.

This is a utility script for LibidoMechanica and, like the other parts of that
program, is licensed under the GPL, either version 3 or (at your option) any
later version. See the file LICENSE.md for details.

This script is copyright 2018 by Patrick Mooney.
"""


import bz2, contextlib, glob, os, pickle, shlex, subprocess, time
from typing import List

import generate
from utils import *
import text_generator as tg

import text_handling as th                          # https://github.com/patrick-brian-mooney/python-personal-library/

import searcher                                     # https://github.com/patrick-brian-mooney/python-personal-library/

import patrick_logger                               # https://github.com/patrick-brian-mooney/python-personal-library/
from patrick_logger import log_it
patrick_logger.verbosity_level = 2


tests_performed_cache_loc = os.path.join(generate.home_dir, 'tests_performed.pkl.bz2')
dictionaries_loc = os.path.join(generate.home_dir, 'dictionaries')

tests_performed = dict()        # We'll reassign this pretty soon, assuming there's cached testing data.
proper_nouns = [][:]            # Again, this is about to be reassigned.


try:
    with bz2.open(tests_performed_cache_loc, 'rb') as the_cache:
        tests_performed = pickle.load(the_cache)
except(IOError, pickle.PickleError):
    pass


def load_proper_noun_data() -> List[str]:
    """Reads all known dictionaries; aggregates capitalized words; returns the
    aggregate list. Makes lots of assumptions, including:
      * all files are in utf-8 encoding and end with the extension ".utf-8";
      * a word is capitalized in a dictionary iff it should always be capitalized;
      * dictionaries are all in the DICTIONARIES_LOC directory;
      * all dictionaries have one word per line.
    """
    log_it("Loading known proper nouns ...", 2)
    ret = set()
    for f in [f for f in searcher.get_files_list(dictionaries_loc) if os.path.isfile(f) and f.endswith('utf-8')]:
        with open(f) as dict_file:
            entries = frozenset({i.strip() for i in dict_file.readlines()})     # The set of words in the dictionary
        new = frozenset({l.strip() for l in entries if th.is_capitalized(l)})   # Capitalized words in dictionary
        also_lower = frozenset({i for i in new if i.lower() in entries})
        new = set(new - also_lower)                                             # drop words that are also in the dictionary in lowercase.
        ret |= new
    log_it("  ... done.", 2)
    return ret


proper_nouns = load_proper_noun_data()      # Now we have a list of known proper nouns.


def write_cache() -> None:
    """Write the 'tests performed' cache out to disk."""
    global tests_performed
    with bz2.open(tests_performed_cache_loc, 'wb') as the_cache:
        pickle.dump(tests_performed, the_cache, protocol=1)


@contextlib.contextmanager
def open_cache() -> None:
    """A context manager that ensures the global cache of tests performed is updated
    when the context manager is exited.
    """
    global tests_performed
    yield None
    write_cache()


def prompt_and_confirm(prompt: str) -> bool:
    """Asks the user to confirm an action with Y/N. Treats a blank as a confirmation.
    Returns True (user confirmed the action) or False (user declined).
    """
    try:
        return input("%s? [Y/n]  " % prompt).strip().lower()[0] == "y"
    except IndexError:          # Not even one character of input?
        return True


def get_first_word(line: str) -> str:
    """Get the first word of the string, ignoring and dropping any punctuation."""
    if not line.strip():
        return line.strip()
    without_initial_punc = line[th._find_first_alphanumeric(line):]
    ret = tg.TextGenerator._tokenize_string(without_initial_punc)[0]
    return th.strip_non_alphanumeric(ret)


def check_test_performed(text, test: str) -> bool:
    """Check if TEST has already been performed on TEXT, and whether TEXT has changed
    since that happened.
    """
    global tests_performed
    if os.path.basename(text) in tests_performed:
        if os.path.getmtime(text) > tests_performed[os.path.basename(text)]['time']:        # Text has been modified. Tests need to be re-run.
            del tests_performed[os.path.basename(text)]
            return False
        return test in tests_performed[os.path.basename(text)]['tests']
    return False


def set_test_performed(text, test: str) -> None:
    """Track that TEST has been performed on TEXT."""
    global tests_performed
    if os.path.basename(text) in tests_performed:
        tests_performed[os.path.basename(text)]['tests'] |= { test }
    else:
        tests_performed[os.path.basename(text)] = {'time': time.time(), 'tests': set([test]) }


def decapitalize_beginnings_of_lines(the_poem: List[str], poem_path: str) -> List[str]:
    """Checks each line. If it's capitalized, we check to see if it should be. If it
    is, but shouldn't be, we decapitalize it.
    """
    capitalized_lines = [ l for l in the_poem if len(l.split()) > 0 and th.is_capitalized(l.split()[0]) ]
    textual_lines = [ l for l in the_poem if len(l.strip()) > 0 ]
    log_it("\nPoem: %s:" % shlex.quote(os.path.basename(poem_path)))
    log_it("%d capitalized lines" % len(capitalized_lines))
    log_it("%d non-blank lines" % len(textual_lines))
    log_it("%.4g%% capitalized" % (100 * (len(capitalized_lines) / len(textual_lines))))
    if prompt_and_confirm("Open this file in external editor"):
        subprocess.call('bluefish %s &' % shlex.quote(poem_path), shell=True)
    if prompt_and_confirm("Auto-lowercase lines"):
        saves = set()  # Remember: SAVES is zero-based. Convert to/from 1-based notation when communicating with the user.
        if prompt_and_confirm("Attempt to guess which lines should remain capitalized (and confirm)"):
            previous_textual_line, end_of_previous_line = None, ""
            for count, line in enumerate(the_poem):
                if th.is_capitalized(line):
                    if count == 0:                      # First line of a poem is the first line of a sentence. Capitalize it.
                        saves.add(count)
                    elif (previous_textual_line) and (len(set(end_of_previous_line) & set(tg.sentence_ending_punct))):
                        saves.add(count)    # If the previous line is not blank ... and there's sentence-ending punctuation after the last alphanumeric character on the previous line ...
                    elif line[th._find_first_alphanumeric(line):2 + th._find_first_alphanumeric(line)] == "I ": #FIXME: what about other whitespace?
                        saves.add(count)
                    elif get_first_word(line) in proper_nouns:
                        if get_first_word(line).strip() not in ['A', 'I']:
                            saves.add(count)
                if line:
                    previous_textual_line = line
                    end_of_previous_line = previous_textual_line[1 + th._find_last_alphanumeric(previous_textual_line):]
            log_it("  OK, I suggest keeping the following %d lines capitalized: %s\n" % (len(saves), sorted([i + 1 for i in saves])))
            if not prompt_and_confirm("Accept this decision"):
                if prompt_and_confirm("Re-enter manually"):
                    saves = set()           # User disagrees? Force manual entry
                else:
                    done = False
                    while not done:
                        if prompt_and_confirm("Remove lines from set"):
                            ans = { int(i) - 1 for i in input("Which lines listed would you like to remove from the set? [comma-sep 1-based list]  ").strip().strip(',').strip().split(',') }
                            saves -= ans
                        if prompt_and_confirm("Add lines to set"):
                            ans = { int(i) - 1 for i in input("Which lines listed would you like to add to the set? [comma-sep 1-based list]  ").strip().strip(',').strip().split(',') }
                            saves |= ans
                        log_it("Current lines that will definitely retain current capitalization: %s" % sorted(list(saves)))
                        done = prompt_and_confirm("Is that data correct")
        if not saves:
            saves = { int(i)-1 for i in input("Which lines should remain uppercased? [comma-sep 1-based list]  ").strip().strip(',').strip().split(',') }
        for num, line in enumerate(the_poem):
            if len(line.strip()) > 0:
                if th.is_capitalized(line) and num not in saves:
                    the_poem[num] = th.decapitalize(line)
    else:
        log_it("  OK, skipping that file...")
    return the_poem


def check_file(what_file: str) -> None:
    """Reads WHAT_FILE in, makes a copy of the data, and then validates that the
    original data meets certain expectations. If the data is changed, writes it back
    over the original file.

    Makes a lot of assumptions, including the assumption that all text files are in
    the system default encoding.
    """
    global tests_performed

    log_it("\n\nINFO: checking %s" % os.path.basename(what_file), 3)
    with open(what_file) as the_file:
        original_text = the_file.read()
    the_text = original_text.split('\n')

    # First, perform line-by-line tests.
    if not check_test_performed(what_file, strip_trailing_whitespace):
        the_text = [ l.rstrip() for l in the_text ]      # Drop whitespace at the end of lines
        set_test_performed(what_file, strip_trailing_whitespace)
    else:
        log_it("    Skipping test for trailing whitespace: already performed!", 5)

    if not check_test_performed(what_file, decapitalize_line_beginnings):
        the_text = decapitalize_beginnings_of_lines(the_text, what_file)
        set_test_performed(what_file, decapitalize_line_beginnings)
    else:
        log_it("    Skipping check for beginning-of-line capitalization: already performed!", 5)

    # Now, whole-text tests.
    the_text = '\n'.join(the_text)

    if not check_test_performed(what_file, no_dumb_quotes):
        if '"' in the_text or "'" in the_text:
            the_text = generate.curlify_quotes(the_text, "'", "‘", "’")
            the_text = generate.curlify_quotes(the_text, '"', '“', '”')
            log_it("Curlified quotes!", 2)
        else:
            log_it("No quotes to curlify!", 4)
        set_test_performed(what_file, no_dumb_quotes)
    else:
        log_it("    Skipping check for dumb quotes: already performed!", 5)

    if not check_test_performed(what_file, two_newlines_at_end):
        while not the_text.endswith('\n\n'):
            the_text += "\n"
        while the_text.endswith('\n\n\n'):
            the_text = the_text[:-1]
        set_test_performed(what_file, two_newlines_at_end)
    else:
        log_it("    Skipping check for two end newlines: already performed!", 5)


    if the_text != original_text:
        log_it("  INFO: file changed: %s. Saving ..." % os.path.basename(what_file), 1)
        with open(what_file, 'w') as the_file:
            the_file.write(the_text)

    tests_performed[os.path.basename(what_file)]['time'] = time.time()       # We have verified data up to now. Reflect this.


if __name__ == "__main__":
    with open_cache() as _:
        for count, f in enumerate(sorted(glob.glob(os.path.join(generate.poetry_corpus, '*')))):
            check_file(f)
            if count > 0 and count % 5 == 0:
                write_cache()
    write_cache()
