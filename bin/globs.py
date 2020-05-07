#! /usr/bin/env python3.5
# -*- coding: utf-8 -*-
"""globs.py contains utilities for Patrick Mooney's LibidoMechanica project.
This file is part of the LibidoMechanica scripts, a project that is copyright
2016-20 by Patrick Mooney. It is alpha software and the author releases it
ABSOLUTELY WITHOUT WARRANTY OF ANY KIND. You are welcome to use it under the
terms of the GNU General Public License, either version 3 or (at your option)
any later version. See the file LICENSE.md for details.
"""

import os, string


home_dir = '/LibidoMechanica'

poetry_corpus = os.path.join(home_dir, 'poetry_corpus')
post_archives = os.path.join(home_dir, 'archives')
data_cache_location = os.path.join(home_dir, 'cache')
sharded_cache_location = data_cache_location
similarity_cache_location = os.path.join(home_dir, 'similarity_cache.pkl.bz2')

lock_file_dir = home_dir
running_lock_name = 'running.pid'
updating_lock_name = 'updating.pid'


known_punctuation = string.punctuation + "‘’“”"
open_quotes = ("‘", "“")
close_quotes = ("’", "”")

words_with_initial_apostrophes = ['tis', 'twas', 't is', 't was', 'gainst']


# Names of validation tests
strip_trailing_whitespace = 'strip trailing whitespace'
decapitalize_line_beginnings = "decapitalize beginnings of lines"
two_newlines_at_end = "file ends with two newlines"
no_dumb_quotes = "file contains no dumb quotes"
