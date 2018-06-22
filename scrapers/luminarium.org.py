#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Downloads selected poems from luminarium.org and saves them to a text file.

This is just a quick hack, but it's licensed under the GNU GPL, either version
3 or, at your option, any later version; see the file LICENSE.md for details.

This code is copyright 2017 by Patrick Mooney.
"""


import html, os, requests, xml

import bs4                      # https://www.crummy.com/software/BeautifulSoup/
from bs4 import BeautifulSoup

import bleach                   # https://pypi.python.org/pypi/bleach

import text_handling as th      # From https://github.com/patrick-brian-mooney/python-personal-library


files_to_download = '/LibidoMechanica/scrapers/luminarium.org/urls.list'
failed_poems = [][:]


def remove_tags(text):
    """Thanks to http://stackoverflow.com/a/35880312"""
    return bleach.clean(text, tags=[], attributes={}, styles=[], strip=True)


with open(files_to_download) as list_file:
    url_list = [ f.strip() for f in list_file.readlines() ]

for which_url in sorted(url_list):
    try:
        if not len(which_url.strip()): continue                     # Don't bother trying to process blank lines

        print("Processing '%s' ...  " % which_url, end='')
        page = requests.get(which_url)
        soup = BeautifulSoup(page.content, 'html.parser')

        html_title = soup.find('title').decode()
        try:
            *_, poem_author, poem_title = [t.strip().strip('.').strip() for t in remove_tags(html_title).split(':')]
        except ValueError:
            poem_author, poem_title = [t.strip().strip('.').strip() for t in remove_tags(html_title).split(':')]
        poem_filename = '%s/%s: "%s"' % (os.path.dirname(files_to_download), poem_author.strip(), poem_title.strip())

        poem_with_cruft = soup.body.find_all('td')[1].table.tr.td.text

        # OK, do any HTML preprocessing we need to do.      
        # Is any of the below even necessary? Aren't we getting plain text already from this site?
        # Might as well run through basic cleanup in case there are inconsistencies in the format of the site.  
        poem_with_cruft = th.multi_replace(poem_with_cruft, [['<br>', '\n'],
                                                             ['<br />', '\n'],
                                                             ['<br/>', '\n'],])
        poem_with_cruft = poem_with_cruft.replace('<div>', '<div>\n\n')
        poem_with_cruft = poem_with_cruft.replace('<p>', '<p>\n\n')
        
        plain_text_poem = remove_tags(poem_with_cruft)
        plain_text_poem = html.unescape(plain_text_poem)
        with open(poem_filename.strip(), mode="w") as poem_file:
            poem_file.write(plain_text_poem)
            print('done!')

    except Exception as e:
        print("... failed! The system said:\n%s" %e)
        failed_poems += [ which_url ]

print("\n\nAll URLs processed! Hooray!\n")
if len(failed_poems):
    with open('%s/failed.url' % os.path.dirname(files_to_download), mode='w') as failed_file:
        failed_file.writelines(['%s\n' % l for l in failed_poems])
    print('\n ...  but %d failed URLs written to %s/failed.url. Alas.\n\n' % (len(failed_poems), os.path.dirname(files_to_download)))
