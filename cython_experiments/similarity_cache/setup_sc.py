#!/usr/bin/env python3.5
"""Quick setup.py-style script to compile similarity_cache.pyx with Cython.
Not needed unless you're trying to compile with Cython.

Copyright 2020 by Patrick Mooney. You are free to use this script for any
purpose whatsoever.
"""


from setuptools import setup
from Cython.Build import cythonize

setup(
    name='similarity cache',
    ext_modules=cythonize("similarity_cache.pyx"),
    zip_safe=False,
)
