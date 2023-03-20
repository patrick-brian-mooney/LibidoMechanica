#! /usr/bin/env python3.5
# -*- coding: utf-8 -*-
# cython: language_level=3


__doc__ = """similarity_cache.pyx is a utility class for Patrick Mooney's LibidoMechanica
project. It provides a class that tracks calculated "similarity" between texts
in the poetry corpus. This is an expensive calculation whose value needs to be
known repeatedly over multiple runs of the script, so it's saved once made.

This file is part of the LibidoMechanica scripts, a project that is copyright
2016-23 by Patrick Mooney. It is alpha software and the author releases it
ABSOLUTELY WITHOUT WARRANTY OF ANY KIND. You are welcome to use it under the
terms of the GNU General Public License, either version 3 or (at your option)
any later version. See the file LICENSE.md for details.

A short log of optimization attempts
* On 7-8 May 2020, a full similarity cache was built for 1720 source texts in
  377 minutes using was a pure-Python version of this module. The similarity
  cache occupied 39.1 MB.
* AN 8 May repeat with no changes at all except in the extension of the file
  from .py to .pyx resulted in no appreciable improvement: 377 minutes on the
  same poetry corpus with a resulting cache of 39.1 MB.
* Replacing a Python class that winds up taking two attributes with a cdef
  struct that houses the same two attributes on 8 May reduced the run time to
  317 minutes and produced a cache of 19.9 MB. As perhaps a more useful metric,
  it takes right about 0.5GB of RAM to load on my current laptop. Also, Cython
  seems to be internally converting these structs to dicts somewhere, perhaps
  when stuffing them into a dict? I have not yet looked into this.

  Also, since get_source_texts and the two functions it calls have also been
  moved to this .pyx file, it's become blindingly fast, which is a nice benefit.

  Another experiment running under PyPy showed that generate.py runs
  substantially slower and takes more memory than under CPython.
* cdef'ing some functions and variables, then re-running on 8-9 May,
  *increased* the run time to 344 minutes without changing the size of the
  cache produced. This is likely a result of other things running in the
  background overnight, though.

"""


import abc
import argparse
import contextlib
import bz2
import functools
import glob

from pathlib import Path

import pickle
import random
import shelve
import sys
import time
import typing


import pid                                              # https://pypi.python.org/pypi/pid/

try:
    import cython                   # For at least one thing, we need to check if we're compiled with Cython
except ImportError:                 # But Cython may not be installed.
    import argparse                 # If not, create a fake Cython object with enough info to be useful.
    cython = argparse.Namespace()
    cython.compiled = False

from globs import *
import text_generator as tg
import poetry_generator as pg


DISALLOW_ABCS = False       # If running in pure-Python mode and this is False, enforce ABC-based constraints on


class Logger:
    verbosity = 1

    def log_it(self, what: str,
               min_level: int = 1) -> None:
        if self.verbosity >= min_level:
            print(what)


logger = Logger()
log_it = logger.log_it


@functools.lru_cache(maxsize=8)
def get_mappings(f: typing.Union[str, Path],
                 markov_length: int) -> tg.MarkovChainTextModel:
    """Trains a generator, then returns the calculated mappings."""
    log_it("get_mappings() called for file %s" % f, 5)
    return pg.PoemGenerator(training_texts=[f], markov_length=markov_length).chains.mapping


@functools.lru_cache(maxsize=2048)
def _comparative_form(what: typing.Union[str, Path]) -> str:
    """Get the standard form of a text's name for indexing lookup purposes."""
    return os.path.basename(str(what).strip()).lower()


def _all_poems_in_comparative_form() -> typing.Dict[str, typing.Union[str, Path]]:
    """Get a dictionary mapping the comparative form of all poems currently in the
    corpus to their full actual filename.
    """
    return {_comparative_form(p): os.path.abspath(p) for p in glob.glob(os.path.join(poetry_corpus, '*'))}


def calculate_overlap(one: typing.Dict[typing.Tuple[str, str], typing.Dict[str, float]],
                      two: typing.Dict[typing.Tuple[str, str], typing.Dict[str, float]]) -> float :
    """Returns the percentage of chains in dictionary ONE that are also in
    dictionary TWO.
    """
    overlap_count = 0
    for which_chain in one.keys():
        if which_chain in two: overlap_count += 1
    return overlap_count / len(one)


abstract_metaclass_parent = object if (DISALLOW_ABCS or cython.compiled) else abc.ABC


class SimilarityEntry:
    when = 0.0
    similarity = 0.0


class AbstractSimilarityCache(abstract_metaclass_parent):
    """This class is the object that manages the global cache of text similarities.
    Most subclasses of this class have not managed to improve on its performance,
    nor on requirements while maintaining decent performance (though see
    ChainMapSimilarityCache for a good alternative implementation).

    The object's internal data cache is a dictionary:
        { (text_name_one, text_name_two):         (a tuple)
              { 'when':,                          (a datetime: when the calculation was made)
                'similarity':                     (a value between 0 and 1, rather heavily weighted toward zero)
                  }
             }
    """
    @staticmethod
    def _key_from_texts(first: typing.Union[str, Path],
                        second: typing.Union[str, Path]) -> typing.Tuple[str, str]:
        """Given texts ONE and TWO, produce a hashable key that can be used to index the
        _data dictionary.
        """
        one, two = _comparative_form(first), _comparative_form(second)
        if one > two:
            one, two = two, one
        return (one, two)

    @abc.abstractmethod
    def __init__(self, cache_file: typing.Union[Path, str]=similarity_cache_location):
        """This method needs to use the CACHE_FILE path to set up self._data to refer
        to a dict-like Hasmap that allows lookups of similarity data; to set the
        self._dirty flag to be initially False; and to do any other setup that's
        required by the implementation.
        """
        self._data = None
        self._dirty = False
        self._cache_file = cache_file

    def close_cache(self) -> None:
        """Do whatever needs to be done to clean up the cache when it's no longer going to
        be used. For instance, if the cache has a database connection, that connection
        may need to be closed, or changes may need to be written to disk.

        Most caches need do nothing here.
        """
        pass

    def __repr__(self) -> str:
        cname = self.__class__.__name__
        try:
            return f"< {cname}, with {len(self._data)} results cached >"
        except AttributeError:
            return f"< {cname} (not fully initialized: no data attached) >"
        except BaseException as err:
            return f"< {cname} (unknown state because {err}) >"

    @abc.abstractmethod
    def flush_cache(self):
        """Writes the textual similarity cache to disk, if self._dirty is True. If
        self._dirty is False, it silently returns without doing anything.
        """

    def _store_data(self, one: typing.Union[str, Path],
                     two: typing.Union[str, Path],
                     similarity: float):
        """Store the SIMILARITY (a float between 0 and 1, weighted toward zero)
        between texts ONE and TWO in the cache.
        """
        entry = argparse.Namespace()
        entry.when = time.time()
        entry.similarity = similarity
        key = self._key_from_texts(one, two)
        self._data[key] = entry

    def calculate_similarity(self, one: typing.Union[str, Path],
                             two: typing.Union[str, Path],
                             markov_length: int=5):
        """Come up with a score evaluating how similar the two texts are to each other.
        This actually means, more specifically, "the product of (a) the percentage of
        chains in the set of chains of length MARKOV_LENGTH constructed from text ONE
        that are also in text TWO; multiplied by (b) the percentage of chains of
        length MARKOV_LENGTH constructed from text TWO that are also in chains
        constructed from text ONE. That is, it's a float between 0 and 1, heavily
        weighted toward zero.

        This routine also caches the calculated result in the global similarity cache.
        It's a comparatively expensive calculation to make, so we store the results.
        """
        log_it("calculate_similarity() called for: %s" % [one, two], 5)
        if one == two:
            return 1.0                      # Well, that's easy.
        chains_one = get_mappings(one, markov_length)
        chains_two = get_mappings(two, markov_length)
        ret = calculate_overlap(chains_one, chains_two) * calculate_overlap(chains_two, chains_one)
        self._store_data(one, two, ret)
        self._dirty = True
        return ret

    def get_similarity(self, one: typing.Union[str, Path],
                       two: typing.Union[str, Path]):
        """Checks to see if the similarity between ONE and TWO is already known. If it is,
        returns that similarity. Otherwise, calculates the similarity and stores it in
        the global similarity cache, which is written at the end of the script's run.

        In short, this function takes advantage of the memoization of
        calculate_similarity, also taking advantage of the fact that
        calculate_similarity(A, B) = calculate_similarity(B, A). It also watches to make
        sure that neither of the texts involved has been changed since the calculation
        was initially made. If either has, it re-performs the calculation and stores
        the updated result in the cache.

        Note that calculate_similarity() itself stores the results of the function. This
        function only takes advantage of the stored values.
        """
        # Index in lexicographical order, by actual filename, after dropping path
        key = self._key_from_texts(one, two)
        log_it(f"get_similarity() called for files: {key}", 5)

        if key in self._data:                       # If it's in the cache, and the data isn't stale ...
            entry = self._data[key]
            if entry.when < os.path.getmtime(one):
                log_it("  ... but cached data is stale relative to %s !" % one, 6)
                return self.calculate_similarity(one, two)
            if entry.when < os.path.getmtime(two):
                log_it("  ... but cached data is stale relative to %s !" % two, 6)
                return self.calculate_similarity(one, two)
            log_it(" ... returning cached value!", 6)
            return entry.similarity

        log_it(" ... not found in cache! Calculating and cacheing ...", 6)
        return self.calculate_similarity(one, two)

    def build_cache(self):
        """Sequentially go through the corpus, text by text, forcing comparisons to all
        other texts and cacheing the results, to make sure the cache is fully
        populated. Periodically, it dumps the results to disk by updating the on-disk
        cache, so that not all of the calculation results are lost if the run is
        interrupted.

        This method takes a VERY long time to run if starting from an empty cache with
        many source texts in the corpus. The cache CAN OF COURSE be allowed to
        populate itself across multiple runs: this particular method is completely
        unneeded for anything other than some testing applications.
        """
        print("Building cache ...")
        for i, first_text in enumerate(sorted(glob.glob(os.path.join(poetry_corpus, '*')))):
            if i % 5 == 0:
                print("  We've performed full calculations for %d texts!" % i)
                if i % 20 == 0:
                    while self._dirty:
                        try:
                            with pid.PidFile(piddir=home_dir):
                                self.flush_cache()                          # Note that success clears self._dirty.
                        except pid.PidFileError:
                            time.sleep(5)                                   # In use? Wait and try again.
            for j, second_text in enumerate(sorted(glob.glob(os.path.join(poetry_corpus, '*')))):
                log_it("About to compare %s to %s ..." % (os.path.basename(first_text), os.path.basename(second_text)), 6)
                _ = self.get_similarity(first_text, second_text)

    def clean_cache(self):
        """Run through the cache, checking for problems and fixing them. Work on a copy
        of the data, then rebind the copy to the original name after cleaning is done.
        """
        log_it("Pruning similiarity cache ...", 1)
        if (not self._data) or (len(self._data) == 0):
            log_it("Nothing to prune: Cache is empty!", 1)
            return

        pruned = {k: v for k, v in self._data.items()}
        comp_form_corpus = _all_poems_in_comparative_form()
        for count, tup in enumerate(self._data):
            try:
                one, two = eval(tup)        # turn string into tuple and unpack
                if count % 1000 == 0:
                    log_it("We're on entry # %d: that's %.3f %% done!" % (count, (100 * count/len(self._data))), 2)
                assert one in comp_form_corpus, "'%s' does not exist!" % one
                assert two in comp_form_corpus, "'%s' does not exist!" % two
                assert one <= two, "%s and %s are mis-ordered!" % (one, two)
                assert self._data[self._key_from_texts(one, two)].when >= os.path.getmtime(comp_form_corpus[one]), f"data for '{one}' is stale!"
                assert self._data[self._key_from_texts(one, two)].when >= os.path.getmtime(comp_form_corpus[two]), f"data for '{two}' is stale!"
                _ = int(self._data[self._key_from_texts(one, two)].when)
                _ = self._data[self._key_from_texts(one, two)].similarity
            except (AssertionError, ValueError, KeyError, AttributeError) as err:
                log_it(f"Removing entry: ({one}, {two})    -- because: {err}", 3)
                del pruned[self._key_from_texts(one, two)]
                self._dirty = True
            except BaseException as err:
                log_it("Unhandled error: {err! Leaving data in place", 3)

        removed = len(self._data) - len(pruned)
        if self._data:
            print(f"Removed {removed} entries; that's { 100 * removed/len(self._data)}%!")
        else:
            print("All entries removed!")
        self._data = pruned
        self.flush_cache()


class BasicSimilarityCache(AbstractSimilarityCache):
    def __init__(self, cache_file: typing.Union[Path, str] = similarity_cache_location):
        log_it("Loading cached similarity data ...", 3)
        try:
            with bz2.open(cache_file, "rb") as pickled_file:
                self._data = pickle.load(pickled_file)
        except (OSError, EOFError, AttributeError, pickle.PicklingError) as err:
            log_it(f"WARNING! Unable to load cached similarity data because {err}.\nUsing empty cache ...", 2)
            self._data = dict()
        log_it("    ... similarity cache loaded! Finalizing data structures.")
        self._dirty = False
        self._cache_file = cache_file

    def flush_cache(self):
        """Writes the textual similarity cache to disk, if self._dirty is True. If
        self._dirty is False, it silently returns without doing anything.

        #FIXME: This function does not do any file locking; there's nothing preventing
        multiple attempts to update the cache at the same time. The convention for
        reducing this problem is that any code, before calling flush_cache(), must
        acquire a PidFile lock distinct from the one the main script acquires before
        beginning its run. Code that needs to write to the cache needs to repeatedly
        attempt to acquire this lock, waiting in between failed attempts, until it is
        able to do so. See .build_cache() for an example.

        In fact, we should be using some sort of real database-like thing, because the
        overhead of keeping all this data in memory is certain to grow quite large.
        """
        if not self._dirty:
            log_it("Skipping cache update: no changes made!", 4)
            return
        log_it("Updating similarity data cache on disk ...", 3)
        with bz2.open(self._cache_file, 'wb') as pickled_file:
            pickle.dump(self._data, pickled_file, protocol=1)
        log_it(" ... updated!", 3)
        self._dirty = False


class DBSimilarityCache(AbstractSimilarityCache):
    @staticmethod
    def _key_from_texts(first: typing.Union[str, Path],
                        second: typing.Union[str, Path]) -> str:
        return str(AbstractSimilarityCache._key_from_texts(first, second))

    def __init__(self, cache_file: typing.Union[Path, str] = similarity_cache_location):
        assert isinstance(cache_file, (Path, str))
        if isinstance(cache_file, str):
            cache_file = Path(cache_file)
        while cache_file != (cache_file.parent / cache_file.stem):
            cache_file = cache_file.parent / cache_file.stem

        log_it(f"Loading cached similarity data from {cache_file} ...", 3)
        try:
            self._data = shelve.open(str(cache_file), "c")
        except (OSError, EOFError, AttributeError) as err:
            log_it(f"WARNING! Unable to load cached similarity data because {err}. Using empty cache ...", 1)
            self._data = dict()
        log_it("    ... cache ready! Finalizing data structures.")
        self._dirty = False
        self._cache_file = cache_file

    def close_cache(self, silent: bool = False) -> None:
        if self._data is not None:
            try:
                self._data.close()
            except (AttributeError,):   # No .close() method because we're using a fallback dict? Oh well.
                pass
            if not silent:
                log_it("    ... cache closed!", 3)
            self._data = None
        else:
            log_it("Warning! Attempt to close already_closed cache!")

    def flush_cache(self) -> None:
        """Force-flush cache data by closing and opening the cache's shelve file..
        """
        c_file = self._cache_file
        self.close_cache(silent=True)
        self._data = shelve.open(str(c_file), "c")
        self._dirty = False


# This variable is an alias that points at whatever the current cache implementation is.
CurrentSimilarityCache = DBSimilarityCache


@contextlib.contextmanager
def open_cache():
    """A context manager that returns the persistent similarity cache and closes it,
    updating it if necessary, when it's done being used. This function repeatedly
    attempts to acquire exclusive access to the cache until it is successful at
    doing so, checking the updating_lock_name lock until it can acquire it.
    """
    opened = False
    while not opened:
        try:
            with pid.PidFile(piddir=lock_file_dir, pidname=updating_lock_name):
                similarity_cache = CurrentSimilarityCache()
                yield similarity_cache
                opened = True
                if similarity_cache._dirty:
                    log_it("DEBUG: The open_cache() context manager is about to flush the similarity cache.", 2)
                    similarity_cache.flush_cache()
        except pid.PidFileError:                        # Already in use? Wait and try again.
            time.sleep(10)


def clean_cache():
    """Clean out the existing file similarity cache, then quit."""
    with open_cache() as similarity_cache:
        similarity_cache.clean_cache()
    sys.exit(0)


def build_cache():
    """Clean out the existing file similarity cache, then make sure it's fully populated."""
    with open_cache() as similarity_cache:
        similarity_cache.clean_cache()
        similarity_cache.build_cache()
    sys.exit(0)


oldmethod = False                # Set to True when debugging to use the (much faster) old method as a fallback.


def old_selection_method(available: typing.Iterable,
                         post_data: dict
                         ) -> typing.Iterable:       # FIXME: iterable of what?
    """This is the original selection method, which merely picks a set of texts at
    random from the corpus. It is fast, but makes no attempt to select texts that
    are similar to each other. This method of picking training texts often produces
    poems that "feel disjointed" and that contain comparatively longer sections of
    continuous letters from a single source text.

    AVAILABLE is the complete list of available texts.
    """
    log_it(" ... according to the old (pure random choice) method")
    post_data['tags'] += ['old textual selection method']
    return random.sample(available, random.randint(75, 150))


def new_selection_method(available: typing.List,
                         similarity_cache: typing.Type[BasicSimilarityCache],
                         post_data: dict,
                         seed_texts: typing.Optional[typing.Iterable] = None        # FIXME: iterable of what?
                         ) -> typing.Iterable:                                      # FIXME: iterable of what?
    """The "new method" for choosing source texts involves picking a small number of
    seed texts, either specified as SEED_TEXTS, or, if that parameter is not filled,
    completely at random, then going through and adding to this small corpus by
    looking for "sufficiently similar" texts to texts already in the corpus.
    "Similarity" is here defined as "having a comparatively high number of
    overlapping chains" as the text it's being compared to. A text has similarity
    1.0 when compared to itself, and similarity 0.0 when it is compared to a text
    that generates no chains in common with it (something in a different script,
    say). Typically, two poems in English chosen more or less at random will have a
    similarity score in the range of .015 to .07 or so.

    Given the initial seed set, then, each poem not already in the set is considered
    sequentially. "Considered" here means that each poem in the already-selected
    corpus is given a chance to "grab" the poem under consideration; the more
    similar the two poems are, the more likely the already-in-the-corpus poem is to
    "grab" the new poem. This process repeats until "there are enough" poems in the
    training corpus.

    This is a slow process: it can take several minutes even on my faster computer.
    Because the similarity calculations are comparatively slow, but many of them
    must be performed to choose a set of training poems, the results of the
    similarity calculations are stored in a persistent cache of similarity-
    calculation results between runs.

    AVAILABLE is the complete list of poems in the corpus.
    SIMILARITY_CACHE is the already-loaded BasicSimilarityCache object.
    """
    assert isinstance(available, list), f"ERROR! the AVAILABLE parameter to new_selection_method() should a list, but is instead a {type(available).__name__} !!!"

    if not seed_texts:
        ret = random.sample(available, random.randint(3, 7))  # Seed the pot with several random source texts.
        seed_texts = [os.path.basename(i) for i in ret]
    else:
        ret = seed_texts
    post_data['seed poems'] = seed_texts

    available = set(available)
    for i in ret:
        available.discard(i)  # Make sure already-chosen texts are not chosen again.
    available = list(available)

    done, candidates = False, 0
    announced, last_count = set(), 0
    while not done:
        candidates += 1
        if not available:
            available = [f for f in glob.glob(os.path.join(poetry_corpus, '*')) if not os.path.isdir(f) and f not in ret]   # Refill the list of options if we've rejected them all.
        current_choice = random.choice(available)
        available.remove(current_choice)
        changed = False
        for i in ret:  # Give each already-chosen text a chance to "claim" the new one
            if random.random() < (similarity_cache.get_similarity(i, current_choice) / len(ret)):
                ret += [current_choice]
                changed = True
                break
        if candidates > 10000 and len(ret) >= 75:
            done = True
        if candidates % 5 == 0:
            print("    ... %d selection candidates" % candidates)
            if changed:
                if (1 - random.random() ** 4.5) < ((len(ret) - 100) / 150):
                    done = True
        if candidates % 25 == 0:
            if len(ret) > last_count:
                print("  ... %d selected texts in %d candidates. New: %s" % (len(ret), candidates, {os.path.basename(f) for f in set(ret) ^ announced}))
                announced, last_count = set(ret), len(ret)
            else:
                print("  ... %d selected texts in %d candidates" % (len(ret), candidates))
        if candidates % 1000 == 0:
            if similarity_cache._dirty:
                similarity_cache.flush_cache()
    post_data["rejected training texts"] = candidates - len(ret)
    if similarity_cache._dirty: similarity_cache.flush_cache()
    return ret


def get_source_texts(similarity_cache: typing.Union[typing.Type[BasicSimilarityCache], None],
                     post_data: dict):
    """Return a list of partially randomly selected texts to serve as the source texts
    for the poem we're writing. There are currently two textual selection methods,
    called "the old method" and "the new method." Each is documented in its own
    function docstring.
    """
    log_it("Choosing source texts")
    available = [f for f in glob.glob(os.path.join(poetry_corpus, '*')) if not os.path.isdir(f)]
    if not similarity_cache:
        return old_selection_method(available, post_data)
    else:
        return new_selection_method(available, similarity_cache, post_data)


if __name__ == "__main__":
    # First: if command-line args are used, just clean, or build, then quit.

    # These two options are mostly useful for testing, because they produce caches that are hard for the main script
    # to understand! (Module-relative imports don't work right.) Instead, to actually generatoe the cache,
    # run the top-level generate.py script with --clean or --build.
    if len(sys.argv) == 2:
        if sys.argv[1] in ['--clean', '-c']:
            clean_cache()
        elif sys.argv[1] in ['--build', '-b']:
            build_cache()
        else:
            print("ERROR: the only accepted command-line arguments are -c [--clean] and -b [--build]!")
            sys.exit(1)

    if len(sys.argv) > 2:
        print("ERROR: unable to understand command-line arguments!")
        sys.exit(1)

    # Otherwise, use this as a debugging harness, for walking through in an IDE.
    c = DBSimilarityCache()
    print(c)
    c.clean_cache()
    c.build_cache()

    import random
    for i in range(5000):
        a, b = random.choice(glob.glob(os.path.join(poetry_corpus, '*'))), random.choice(glob.glob(os.path.join(poetry_corpus, '*')))
        print(f"Similarity between %s and %s is: %.4f" % (os.path.basename(a), os.path.basename(b), c.get_similarity(a,b)))

    c.close_cache()
