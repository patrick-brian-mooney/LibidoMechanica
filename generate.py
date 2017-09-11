#! /usr/bin/env python3
"""Script to post to LibidoMechanica.tumblr.com. Really rough sketch of a
script here. Not really meant for public use. Based on my earlier script to
do a similar job for the AutoLovecraft Tumblr account; see
https://github.com/patrick-brian-mooney/AutoLovecraft for more info.

The fact that there isn't real documentation here is intentionally meant to
reinforce that this is a rough draft not meant for public use. Use at your own
risk. My hope is that this script is helpful, but I explicitly disclaim ANY
RESPONSIBILITY for the behavior of this piece of work. If you don't have the
expertise to evaluate its risks, it's not for you. To put it more bluntly: THIS
SOFTWARE IS OFFERED WITHOUT WARRANTY OF ANY KIND AT ALL.

Nevertheless, if you want to adapt it, this script is licensed under the GNU
GPL, either version 3, or (at your option) any later version; see the file
LICENSE.md for more details.
"""

import bz2, datetime, functools, glob, json, os, pprint, random, re, string

import patrick_logger                                   # From https://github.com/patrick-brian-mooney/personal-library
from patrick_logger import log_it
import social_media                                     # From https://github.com/patrick-brian-mooney/personal-library
from social_media_auth import libidomechanica_client    # Unshared file that contains authentication constants

import poetry_generator as pg                           # https://github.com/patrick-brian-mooney/markov-sentence-generator

import text_handling as th                              # From https://github.com/patrick-brian-mooney/personal-library


patrick_logger.verbosity_level = 2

poetry_corpus = '/LibidoMechanica/poetry_corpus'
post_archives = '/LibidoMechanica/archives'


known_punctuation = string.punctuation + "‘’“”"

normalization_strategy, stanza_length = None, None


def print_usage():    # Note that, currently, nothing calls this.
    """Print the docstring as a usage message to stdout"""
    log_it("INFO: print_usage() was called")
    print(__doc__)

def get_title(the_poem):
    """Get a title for the poem. There are several title-generating algorithms; this
    function picks one at random.
    """
    log_it("INFO: getting a title for the poem")
    possible_titles = [
      lambda: "Untitled Poem # %d" % (1 + len(glob.glob(post_archives + '/*Untitled*'))),
      lambda: "Untitled Composition # %d" % (1 + len(glob.glob(post_archives + '/*Untitled*'))),
      lambda: "Untitled # %d" % (1 + len(glob.glob(post_archives + '/*Untitled*'))),
      lambda: "Untitled ('%s')" % th.strip_leading_and_trailing_punctuation(the_poem.split('\n')[0]).strip(),
      lambda: "Untitled ('%s')" % th.strip_leading_and_trailing_punctuation(the_poem.split('\n')[0]).strip(),
      lambda: "‘%s’" % th.strip_leading_and_trailing_punctuation(the_poem.strip().split('\n')[0]).strip(),  # First line, in quotes
      lambda: "‘%s’" % th.strip_leading_and_trailing_punctuation(the_poem.strip().split('\n')[0]).strip(),  # First line, in quotes
      lambda: "‘%s’" % th.strip_leading_and_trailing_punctuation(the_poem.strip().split('\n')[0]).strip(),  # First line, in quotes
      lambda: "‘%s’" % th.strip_leading_and_trailing_punctuation(the_poem.strip().split('\n')[0]).strip(),  # First line, in quotes
      lambda: "‘%s’" % th.strip_leading_and_trailing_punctuation(the_poem.split('\n')[random.randint(1,4)-1]).strip(),
      lambda: "‘%s’" % th.strip_leading_and_trailing_punctuation(the_poem.split('\n')[random.randint(1,4)-1]).strip(),
      lambda: "‘%s’" % th.strip_leading_and_trailing_punctuation(the_poem.split('\n')[random.randint(1,4)-1]).strip(),
      lambda: th.strip_leading_and_trailing_punctuation(genny.gen_text(sentences_desired=1).split('\n')[0].strip()),  # New 'sentence'
      lambda: th.strip_leading_and_trailing_punctuation(genny.gen_text(sentences_desired=1).split('\n')[0].strip()),  # New 'sentence'
      lambda: th.strip_leading_and_trailing_punctuation(genny.gen_text(sentences_desired=1).split('\n')[0].strip()),  # New 'sentence'
    ]
    title = random.choice(possible_titles)()
    if len(title) < 3:                  # Try again, recursively.
        title = get_title(the_poem)
    while len(title) > 90:
        words = title.split()
        title = ' '.join(words[:random.randint(3, min(12, len(words)))])
    if title.startswith('‘') and not title.endswith('’'):       # The shortening procedure above might have stripped the closing quote
        title = title + '’'
    if '(‘' in title and not '’)' in title:
        title = title + '’)'
    return title


def strip_invalid_chars(the_poem):
    """Some characters appear in the training texts but are characters that, I am
    declaring by fiat, should not make it into the final generated poems at all.
    The underscore is a good example of characters in this class. This function
    takes an entire poem as input (THE_POEM) and returns a poem entirely
    stripped of all such characters.
    """
    log_it("INFO: stripping invalid characters from the poem")
    invalids = ['_', '*']
    return ''.join([s for s in the_poem if not s in invalids])

def curlify_quotes(the_poem, straight_quote, opening_quote, closing_quote):
    """Goes through THE_POEM (a string) and looks for instances of STRAIGHT_QUOTE
    (a single-character string). When it finds these instances, it substitutes
    OPENING_QUOTE or CLOSING_QUOTE for them, trying to make good decisions about
    which of those substitutions is appropriate.

    IMPORTANT CAVEAT: this routine iterates over THE_POEM, making in-place
    changes at locations determined via an initial scan. This means that
    OPENING_QUOTE and CLOSING_QUOTE **absolutely must** have the same len() as
    STRAIGHT_QUOTE, or else weird things will happen. This should not be a
    problem with standard English quotes under Python 3.X; but it may fail under
    non-Roman scripts, in odd edge cases, or if the function is used to try to
    do something other than curlify quotes.

    NOT FULLY TESTED, but I'm going to bed.
    """
    log_it("INFO: curlify_quotes() called to differentiate %s (%d) into %s and %s" % (straight_quote, the_poem.count(straight_quote), opening_quote, closing_quote), 2)
    assert len(straight_quote) == 1, "Quote characters passed to curify_quotes() must be one-character strings"
    assert len(opening_quote) == 1, "Quote characters passed to curify_quotes() must be one-character strings"
    assert len(closing_quote) == 1, "Quote characters passed to curify_quotes() must be one-character strings"
    index = 0
    while index < len(the_poem):
        index = the_poem.find(straight_quote, index)
        if index == -1:
            break       # We're done.
        if index == 0:                                                      # Is it the first character of the poem?
            the_poem = opening_quote + the_poem[1:]
        elif index == len(the_poem):                                        # Is it the last character of the poem?
            the_poem = the_poem[:-1] + closing_quote
        elif the_poem[index-1].isspace() and the_poem[index+1].isspace():   # Whitespace on both sides? Replace quote with space.
            the_poem = the_poem[:index] + ' ' + the_poem[1+index:]
        elif not the_poem[index-1].isspace():                               # Non-whitespace immediately before quote? It's a closing quote.
            the_poem = the_poem[:index] + closing_quote + the_poem[index+1:]
        elif not the_poem[index+1].isspace():                               # Non-whitespace just after quote? It's an opening quote.
            the_poem = the_poem[:index] + opening_quote + the_poem[index+1:]
        else:                                                               # Quote appears in middle of non-whitespace text ...
            if straight_quote == '"':
                the_poem = the_poem[:index-1] + the_poem[index+1:]                  # Just strip it out.
            elif straight_quote == "'":
                the_poem = the_poem[:index-1] + closing_quote + the+poem[index+1:]  # Make it an apostrophe.
            else:
                raise NotImplementedError                                           # We don't know how to deal with this quote.
    return the_poem

def balance_punctuation(the_poem, opening_char, closing_char):
    """Make sure that paired punctuation (smart quotes, parentheses, brackets) in the
    poem are 'balanced.' If not, it attempts to correct it.
    """
    opening, closing = the_poem.count(opening_char), the_poem.count(closing_char)
    if closing_char == '’':     # Sigh. We have to worry about apostrophes that look like closing single quotes.
        closing -= len(re.findall('[:alnum:]*’[:alnum:]', the_poem))    # Inside a word? It's an apostrophe. Don't count it

    log_it("INFO: Balancing %s and %s (%d/%d)" % (opening_char, closing_char, opening, closing))

    if opening or closing:      # Do nothing if there's no instances of either character
        if opening != closing:  # Do nothing if we already have equal numbers (even if not properly "balanced")
            nesting_level = 0   # How many levels deep are we right now in the punctuation we're tracking?
            indexed_poem = list(the_poem)
            index = 0
            while index <= (len(indexed_poem)-1):
                char = indexed_poem[index]
                next_char = '' if index == len(indexed_poem) -1 else indexed_poem[index + 1]
                last_char = '' if index == 0 else indexed_poem[index - 1]
                if index == (len(indexed_poem)-1) :  # End of the poem?
                    if nesting_level > 0:               # Close any open characters.
                        indexed_poem += [closing_char]
                        nesting_level -= 1
                    index += 1
                elif char == opening_char:          # Opening character?
                    if index == len(indexed_poem):      # Last character is an opening character?
                        indexed_poem.pop(-1)            # Just drop it.
                    else:
                        nesting_level += 1              # We're one level deeper
                        index += 1                          # Move on to next character
                elif char == closing_char:          # Closing character?
                    if (closing_char == '’') and (th.is_alphanumeric(next_char) and th.is_alphanumeric(last_char)):
                        index += 1      # Skip apostrophes in the middle of words
                    else:
                        if nesting_level < 1:               # Are we trying to close something that's not open?
                            indexed_poem.pop(index)         # Just drop the spurious close quote
                        else:
                            if next_char.isspace():             # Avoid non-quote apostrophes in middle of words.
                                nesting_level -= 1          # We're one level less deep
                            index += 1                      # Move on to next character
                elif nesting_level > 0:             # Are we currently in the middle of a bracketed block?
                    if next_char.isspace():             # Is the next character whitespace?
                        if random.random() < (0.001 * nesting_level):   # Low chance of closing the open bracketer
                            indexed_poem.insert(index, closing_char)
                            nesting_level -= 1
                    elif char in ['.', '?', '!'] and next_char.isspace():
                        if random.random() < (0.05 * nesting_level):    # Higher chance of closing the open bracketer
                            indexed_poem.insert(index, closing_char)
                            nesting_level -= 1
                            if random.random() < 0.2:                       # Force new paragraph break?
                                indexed_poem.insert(index + 1, '\n')
                    elif char in known_punctuation and last_char in ['.', '!', '?']:
                        if random.random() < (0.05 * nesting_level):
                            indexed_poem.insert(index, closing_char)
                            nesting_level -= 1
                            if random.random() < 0.2:                       # Force new paragraph break?
                                indexed_poem.insert(index + 1, '\n')
                    elif char == '\n' and next_char == '\n':            # Very high chance of closing on paragraph boundaries
                        if random.random() < (0.4 * nesting_level):
                            indexed_poem.insert(index, closing_char)
                            nesting_level -= 1
                    elif char == '\n':
                        if random.random() < (0.1 * nesting_level):
                            indexed_poem.insert(index, closing_char)
                            nesting_level -= 1
                    index += 1
                else:
                    index += 1
            the_poem = ''.join(indexed_poem)
    log_it("   ... after balancing, there are %d/%d punctuation marks." % (the_poem.count(opening_char), the_poem.count(closing_char)), 2)
    return the_poem

def fix_punctuation(the_poem):
    """Cleans up the punctuation in the poems so that it appears to be more
    'correct.' Since characters are generated randomly based on a frequency
    analysis of which characters are likely to follow the last three to ten
    characters, there's no guarantee that (for instance) parentheses or quotes
    are balanced, because the generator doesn't pay attention to or understand
    larger-scale structures.

    THE_POEM is a string, which is the text of the entire poem; the function
    returns a new, cleaned-up version of the poem passed in.

    NOT YET FULLY IMPLEMENTED.
    """
    log_it("INFO: about to alter punctuation", 2)
    the_poem = strip_invalid_chars(the_poem)
    the_poem = curlify_quotes(the_poem, "'", "‘", "’")
    the_poem = curlify_quotes(the_poem, '"', '“', '”')
    the_poem = balance_punctuation(the_poem, "‘", "’")
    the_poem = balance_punctuation(the_poem,  '“', '”')
    the_poem = balance_punctuation(the_poem,  '(', ')')
    the_poem = balance_punctuation(the_poem,  '[', ']')
    return balance_punctuation(the_poem,  '{', '}')

def do_basic_cleaning(the_poem):
    """Does first-pass elementary cleanup tasks on THE_POEM. Returns the cleaned
    version of THE_POEM.
    """
    log_it("INFO: about to do basic pre-cleaning of poem", 2)
    the_poem = th.multi_replace(the_poem, [[' \n', '\n'], ['\n\?', '?'], ['\n!', '!'],
                                           ['\n"', '\n'], ['\n”', '\n'], ['\n\n\n', '\n\n'],
                                           ['\n" ', '\n"'], ['^" ', '"']]).strip()
    return the_poem

def factors(n):
    """Return a list of the factors of a number. Based on code at
    < https://stackoverflow.com/a/6800214 >.
    """
    assert (int(n) == n and n > 1), "ERROR: factors() called on %s, which is not a positive integer" % n
    return sorted(list(set(functools.reduce(list.__add__, ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))))

def is_prime(n):
    """Return True if N is prime, false otherwise. "Prime" is here defined specifically
    as "has fewer than three factors," which is not quite the same as mathematical
    ... um, primery. Primeness. Anyway, this is intended to be inclusive about edge
    cases that "is it prime?" should often include.
    """
    assert (int(n) == n and n > 1), "ERROR: is_prime() called on %s, which is not a positive integer" % n
    return (len(factors(n)) < 3)

def lines_without_stanza_breaks(the_poem):
    """Returns a *list* of lines from THE_POEM, removing any stanza breaks.
    """
    return [l for l in the_poem.split('\n') if len(l.strip()) > 0]

def total_lines(the_poem):
    """Returns the total number of non-empty lines in the poem."""
    return len(lines_without_stanza_breaks(the_poem))

def reduce_single_lines(the_poem):
    """Takes the poem passed as THE_POEM and goes through it, (mostly) eliminating
    single-line stanzas. Returns the corrected poem.

    #FIXME: does not yet do anything productive.
    """
    stanzas = [l.split('\n') for l in the_poem.split('\n\n')]   # A list of stanzas, each of which is a list of lines
    i = 0
    while i < len(stanzas):
        if len(stanzas[i]) < 3:
            try:                                # Combine it with the next stanza. Probably.
                if random.random() <= 0.85:
                    next_stanza = stanzas.pop(i+1)
                    stanzas[i] += next_stanza
                else: i += 1
            except IndexError:                  # If there is no next stanza, oh well.
                pass
        else:
            i += 1
    return '\n\n'.join(['\n'.join(s) for s in stanzas])

def regularize_stanza_length(the_poem):
    """Reconfigures stanza breaks in THE_POEM so that it has stanzas of a regular
    length. Returns the modified poem with regular stanzas. Tries to choose a
    reasonable stanza length, but (if all else fails) will just produce one long
    undivided poem.
    """
    global stanza_length
    textual_lines = lines_without_stanza_breaks(the_poem)
    num_lines = len(textual_lines)
    # assert (not is_prime(num_lines)), "ERROR: regularize_stanza_length() called for poem with prime number of lines"
    possible_stanza_lengths = factors(num_lines)
    if len([x for x in possible_stanza_lengths if x >= 3]):     # If possible, prefer stanzas at least as long as Dante's in the Divine Comedy.
        possible_stanza_lengths = [x for x in possible_stanza_lengths if x >= 3]
    if len([x for x in possible_stanza_lengths if x <= 16]):    # If possible, choose a stanza length no longer than Meredith's extended sonnets in /Modern Love/.
        possible_stanza_lengths = [x for x in possible_stanza_lengths if x <= 16]
    stanza_length = random.choice(possible_stanza_lengths)
    if stanza_length == 1: stanza_length = num_lines            # 1 long stanza, not many one-line stanzas
    the_poem = ""
    for stanza in range(0, num_lines // stanza_length):         # Iterate over the appropriate # of stanzas
        for line in range(0, stanza_length):
            the_poem += "%s\n" % textual_lines.pop(0)
        the_poem += '\n'                                        # Add stanza break
    return the_poem

def regularize_form(the_poem):
    """Choose one of several strategies to regularize the form of the poem. Note that
    one of these strategies is 'do nothing'.
    """
    global normalization_strategy
    possible_strategies = [
        ('regular stanza length (experimental)', lambda x: regularize_stanza_length(x)),
        ('regular stanza length (experimental)', lambda x: regularize_stanza_length(x)),
        ('regular stanza length (experimental)', lambda x: regularize_stanza_length(x)),
        ('regular stanza length (experimental)', lambda x: regularize_stanza_length(x)),
        (None, lambda x: x),
        ('reduce single lines (experimental)', lambda x: reduce_single_lines(x)),
        ('reduce single lines (experimental)', lambda x: reduce_single_lines(x)),
    ]
    normalization_strategy, normalization_procedure = random.choice(possible_strategies)
    log_it("INFO: form normalization strategy is: %s" % normalization_strategy, 2)
    return normalization_procedure(the_poem)

def do_final_cleaning(the_poem):
    log_it("INFO: about to do final cleaning of poem", 2)
    the_poem = th.multi_replace(the_poem, [[' \n', '\n'],               # Eliminate any spurious end-of-line spaces
                                           ['\n\n\n', '\n\n'],          # ... and any extra line breaks.
                                           [r'\n\)', r')'],             # And line breaks right before ending punctuation
                                           [' \n', '\n'], ['\n\?', '?'], ['\n!', '!'],
                                           ['\n"', '\n'], ['\n”', '\n'], ['\n’', '’'],
                                           ['“\n', '“'], ['"\n', '"'],  # And line breaks right after beginning punctuation
                                          ])
    poem_lines = the_poem.split('\n')
    index = 0
    while index < (len(poem_lines) - 1):                            # Go through, line by line, making any final changes.
        line = poem_lines[index]
        if '  ' in line.strip():                                    # Multiple whitespace in line? Break into multiple lines
            individual_lines = ['  ' + i + '\n' for i in line.split('  ')]
            if len(individual_lines) > 1:
                poem_lines.pop(index)
                individual_lines.reverse()          # Go through the sub-lines backwards,
                for l in individual_lines:
                    poem_lines.insert(index, l)     # ... inserting lines and pushing the line stack up.
            index += len(individual_lines)
        else:
            index += 1
    the_poem = '\n'.join(poem_lines)
    return regularize_form(the_poem)


if __name__ == "__main__":
    # Set up the basic parameters for the run
    sample_texts = random.sample(glob.glob(poetry_corpus + '/*txt'), random.randint(40,100))
    chain_length = random.choice([3, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 8, 8, 9, 10])

    # And add their names to the list of tags, plus track sources of this particular poem
    source_texts = [ os.path.splitext(th.remove_prefix(os.path.basename(t), "Link to ").strip())[0] for t in sample_texts ]
    the_tags = ['poetry', 'automatically generated text', 'Patrick Mooney', 'Markov chains'] + \
               ['Markov chain length: %d' % chain_length, '%d texts' % len(sample_texts) ]

    poem_length = random.randint(4,20)              # in SENTENCES. Not lines.

    log_it("INFO: about to set up and train text generator ...")
    genny = pg.PoemGenerator(name='Libido Mechanica generator', training_texts=sample_texts, markov_length=chain_length)

    log_it("INFO: about to generate poem ...")
    the_poem = genny.gen_text(sentences_desired=poem_length, paragraph_break_probability=0.2)

    the_title = get_title(the_poem)

    log_it("poem generated; title is: %s" % the_title)
    log_it("lines are: \n\n" + the_poem)
    log_it("tags are: %s" % the_tags)

    log_it("INFO: cleaning poem up ...")
    the_poem = do_basic_cleaning(the_poem)
    the_poem = fix_punctuation(the_poem)
    the_poem = do_final_cleaning(the_poem)

    log_it("INFO: HTML-izing poem ...")
    # Force all spaces to be non-breaking spaces
    formatted_poem = th.multi_replace(the_poem, [[' ', '&nbsp;']])
    # Add HTML <br /> to end of every line
    formatted_poem = '\n'.join([line.rstrip() + '<br />' for line in the_poem.split('\n')])
    # Wrap stanzas in <p> ... </p>
    formatted_poem = '\n'.join(['<p>%s</p>' % line for line in formatted_poem.split('<br />\n<br />')])
    # Eliminate extra line breaks at the very beginning of paragraphs
    formatted_poem = th.multi_replace(formatted_poem, [['<p><br />\n', '<p>'], ['<p>\n', '<p>'], ['<p>\n', '<p>']])
    # Pretty-print (for debugging only; doesn't matter for Tumblr upload, but neither does it cause problems)
    formatted_poem = th.multi_replace(formatted_poem, [['<p>\n', '\n<p>']])
    # Prevent all spaces from collapsing; get rid of spurious paragraphs
    formatted_poem = th.multi_replace(formatted_poem, [['<p></p>', ''], ['<p>\n</p>', '']])
    # formatted_poem = "<pre>\n%s\n</pre>" % formatted_poem         # OK, that looks really ugly when it posts.

    log_it('INFO: Attempting to post the content...')
    the_status, the_tumblr_data = social_media.tumblr_text_post(libidomechanica_client, ', '.join(the_tags), the_title, formatted_poem)
    log_it('INFO: the_status is: ' + pprint.pformat(the_status), 2)
    log_it('INFO: the_tumblr_data is: ' + pprint.pformat(the_tumblr_data), 3)

    log_it("INFO: archiving poem and metadata ...")
    post_data = {'title': the_title, 'text': the_poem, 'time': datetime.datetime.now().isoformat() }
    post_data['formatted_text'], post_data['tags'], post_data['sources'] = formatted_poem, the_tags, sorted(source_texts)
    post_data['status_code'], post_data['tumblr_data'] = the_status, the_tumblr_data
    post_data['normalization strategy'], post_data['stanza length'] = normalization_strategy, stanza_length
    archive_name = "%s — %s.json.bz2" % (post_data['time'], the_title)
    with bz2.BZ2File(os.path.join(post_archives, archive_name), mode='wb') as archive_file:
        archive_file.write(json.dumps(post_data, sort_keys=True, indent=3, ensure_ascii=False).encode())

    log_it("INFO: We're done")
