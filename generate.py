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

import random, pprint, glob, os, datetime, json

import patrick_logger                                   # From https://github.com/patrick-brian-mooney/personal-library
import social_media                                     # From https://github.com/patrick-brian-mooney/personal-library
from social_media_auth import libidomechanica_client    # Unshared file that contains authentication constants

import poetry_generator as pg                           # https://github.com/patrick-brian-mooney/markov-sentence-generator

import text_handling as th                              # From https://github.com/patrick-brian-mooney/personal-library


patrick_logger.verbosity_level = 2

poetry_corpus = '/LibidoMechanica/poetry_corpus'
post_archives = '/LibidoMechanica/archives'


def print_usage():    # Note that, currently, nothing calls this.
    """Print the docstring as a usage message to stdout"""
    patrick_logger.log_it("INFO: print_usage() was called")
    print(__doc__)

def get_title(the_poem):
    """Get a title for the poem."""
    return "Untitled Composition # %d" % len(glob.glob(post_archives))

if __name__ == "__main__":
    
    # Set up the basic parameters for the run
    sample_texts = random.sample(glob.glob(poetry_corpus + '/*txt'), random.randint(6,20))
    chain_length = random.randint(3,7)
    
    # And add their names to the list of tags
    the_tags =  [ os.path.splitext(th.remove_prefix(os.path.basename(title), "Link to ").strip())[0] for title in sample_texts ] + \
                ['poetry', 'automatically generated text', 'Patrick Mooney', 'Python', 'Markov chains'] + \
                ['Markov chain length: %d' % chain_length ]
    
    poem_length = random.randint(4,12)              # in SENTENCES. Not lines.
    
    genny = pg.PoemGenerator(name='Libido Mechanica generator', training_texts=sample_texts, markov_length=chain_length)
        
    the_poem = genny.gen_text(sentences_desired=poem_length, paragraph_break_probability=0.2)
    the_title=get_title(the_poem)
    
    formatted_poem = th.multi_replace(the_poem, [[' \n', '\n'],               # Eliminate any spurious end-of-line spaces
                                                 ['\n\n\n', '\n\n']])         # ... and any extra line breaks.
    
    # FIXME: preserve beginning-of-line spacing
    
    # Add HTML <br /> to end of every line
    formatted_poem = '\n'.join([line.rstrip() + '<br />' for line in formatted_poem.split('\n')])
    # Wrap stanzas in <p> ... </p>
    formatted_poem = '\n'.join(['<p>%s</p>' % line for line in formatted_poem.split('<br />\n<br />')])
    # Pretty-print for debugging only
    formatted_poem = th.multi_replace(formatted_poem, [['<p>\n', '\n<p>']])
    # Get rid of spurious paragraphs    
    formatted_poem = th.multi_replace(formatted_poem, [['<p></p>', '']])
    
    patrick_logger.log_it("poem generated; title is; %s" % the_title)
    patrick_logger.log_it("lines are: \n\n" + the_poem)
    patrick_logger.log_it("tags are: %s" % the_tags)
    
    # All right, we're ready. Let's go.
    patrick_logger.log_it('INFO: Attempting to post the content', 2)
    the_status = social_media.tumblr_text_post(libidomechanica_client, ', '.join(the_tags), the_title, the_poem)
    patrick_logger.log_it('INFO: the_status is: ' + pprint.pformat(the_status), 2)
    
    # Archive the generated post
    with open(os.path.join(post_archives, datetime.datetime.now().isoformat() + '.json'), 'w') as archive_file:
        json.dump(the_status, archive_file)
    
    patrick_logger.log_it("INFO: We're done", 1)