#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""utils.py contains utilities for Patrick Mooney's LibidoMechanica project.
This file is part of the LibidoMechanica scripts, a project that is copyright
2016-18 by Patrick Mooney. It is alpha software and the author releases it
ABSOLUTELY WITHOUT WARRANTY OF ANY KIND. You are welcome to use it under the
terms of the GNU General Public License, either version 3 or (at your option)
any later version. See the file LICENSE.md for details.
"""

import os


home_dir = '/LibidoMechanica'

poetry_corpus = os.path.join(home_dir, 'poetry_corpus')
post_archives = os.path.join(home_dir, 'archives')
similarity_cache_location = os.path.join(home_dir, 'similarity_cache.pkl.bz2')

lock_file_dir = home_dir
running_lock_name = 'running.pid'
updating_lock_name = 'updating.pid'
