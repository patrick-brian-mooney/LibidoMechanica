#! /usr/bin/env python3.5
# -*- coding: utf-8 -*-
# cython: language_level=3

__doc__ = """similarity_cache.pyx is a utility class for Patrick Mooney's LibidoMechanica
project. It provides a class that tracks calculated "similarity" between texts
in the poetry corpus. This is an expensive calculation whose value needs to be
known repeatedly over multiple runs of the script, so it's saved once made.

This file is part of the LibidoMechanica scripts, a project that is copyright
2016-20 by Patrick Mooney. It is alpha software and the author releases it
ABSOLUTELY WITHOUT WARRANTY OF ANY KIND. You are welcome to use it under the
terms of the GNU General Public License, either version 3 or (at your option)
any later version. See the file LICENSE.md for details.
"""

import bz2
import collections
import contextlib
import functools
import glob

from pathlib import Path

import pickle
import shlex
import sys
import time
import typing


import pid                                              # https://pypi.python.org/pypi/pid/


from bin.globs import *
import text_generator as tg
import poetry_generator as pg


def log_it(what: str, min_level: int=1): pass           # FIXME! Let's actually hook this up to a logger.


@functools.lru_cache(maxsize=8)
def get_mappings(f: typing.Union[str, Path],
                 markov_length: int) -> tg.MarkovChainTextModel:
    """Trains a generator, then returns the calculated mappings."""
    log_it("get_mappings() called for file %s" % f, 5)
    return pg.PoemGenerator(training_texts=[f], markov_length=markov_length).chains.the_mapping


@functools.lru_cache(maxsize=2048)
def _comparative_form(what: typing.Union[str, Path]) -> str:
    """Get the standard form of a text's name for indexing lookup purposes."""
    return os.path.basename(str(what).strip()).lower()


def _all_poems_in_comparative_form() -> typing.Dict[str, typing.Union[str, Path]]:
    """Get a dictionary mapping the comparative form of all poems currently in the
    corpus to their full actual filename.
    """
    return {_comparative_form(p):os.path.abspath(p) for p in glob.glob(os.path.join(poetry_corpus, '*'))}


cpdef float calculate_overlap(one: typing.Dict[typing.Tuple[str], typing.Dict[str, float]],
                              two: typing.Dict[typing.Tuple[str], typing.Dict[str, float]]):
    """Returns the percentage of chains in dictionary ONE that are also in
    dictionary TWO.
    """
    cdef int overlap_count = 0
    for which_chain in one.keys():
        if which_chain in two: overlap_count += 1
    return overlap_count / len(one)


def _key_from_texts(first: typing.Union[str, Path],
                    second: typing.Union[str, Path]) -> typing.Tuple[str, str]:
        """Given texts ONE and TWO, produce a hashable key that can be used to index the
        _data dictionary.
        """
        cdef str one, two
        one, two = _comparative_form(first), _comparative_form(second)
        if one > two:
            one, two = two, one
        return (one, two)


cdef struct SimilarityEntry:
    float when
    float similarity


cdef class BasicSimilarityCache(object):
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
    def __init__(self, cache_file: typing.Union[Path, str]=similarity_cache_location):
        try:
            with bz2.open(similarity_cache_location, "rb") as pickled_file:
                log_it("Loading cached similarity data ...", 3)
                self._data = pickle.load(pickled_file)
        except (OSError, EOFError, AttributeError, pickle.PicklingError) as err:
            print("WARNING! Unable to load cached similarity data because %s. Using empty cache ..." % err)
            self._data = dict()
        self._dirty = False
        self._cache_file = cache_file

    def __repr__(self) -> str:
        try:
            return "< Basic Textual Similarity Cache, with %d results cached >" % len(self._data)
        except AttributeError:
            return "< Basic Textual Similarity Cache (not fully initialized: no data attached) >"
        except BaseException as err:
            return "< Basic Textual Similarity Cache (unknown state because %s) >" % err

    cpdef flush_cache(self):
        """Writes the textual similarity cache to disk, if self._dirty is True. If
        self._dirty is False, it silently returns without doing anything.

        Or, rather, that's the basic idea. In fact, what it does is reload the version
        of the cache that's currently on disk and updates it with new info instead of
        replacing the one on disk. The reason for this, of course, is that this
        script has become complex enough that it may take more than an hour to run on
        the slow old laptop that hosts it ... and so there may be multiple copies
        running, each of which thinks it has the "master copy" in memory. To help
        ameliorate the potential for race conditions, we update instead of overwriting.

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
        try:
            with bz2.open(self._cache_file, "rb") as pickled_file:
                old = pickle.load(pickled_file)
            old.update(self._data)
            self._data = old
        except (OSError, EOFError, pickle.PicklingError) as err:
            log_it("Not able to update previous data: %s" % err)
        except BaseException as err:
            log_it("Unhandled exception occurred during cache file update!   %s" % err)
        with bz2.open(self._cache_file, 'wb') as pickled_file:
            pickle.dump(self._data, pickled_file, protocol=1)
        log_it(" ... updated!", 3)
        self._dirty = False

    cdef _store_data(self, one: typing.Union[str, Path],
                     two: typing.Union[str, Path],
                     similarity: float):
        """Store the SIMILARITY (a float between 0 and 1, weighted toward zero)
        between texts ONE and TWO in the cache.
        """
        cdef SimilarityEntry entry
        entry.when = time.time()
        entry.similarity = similarity
        key = _key_from_texts(one, two)
        self._data[key] = entry

    cpdef float calculate_similarity(self, one: typing.Union[str, Path],
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

    cpdef float get_similarity(self, one: typing.Union[str, Path],
                               two: typing.Union[str, Path]) except *:
        """Checks to see if the similarity between ONE and TWO is already known. If it is,
        returns that similarity. Otherwise, calculates the similarity and stores it in
        the global similarity cache, which is written at the end of the script's run.

        In short, this function takes advantage of the memoization of
        calculate_similarity, also taking taking advantage of the fact that
        calculate_similarity(A, B) = calculate_similarity(B, A). It also watches to make
        sure that neither of the texts involved has been changed since the calculation
        was initially made. If either has, it re-performs the calculation and stores
        the updated result in the cache.

        Note that calculate_similarity() itself stores the results of the function. This
        function only takes advantage of the stored values.
        """
        # Index in lexicographical order, by actual filename, after dropping path
        key = _key_from_texts(one, two)
        log_it("get_similarity() called for files: %s" % list(key), 5)

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
        log_it("Building cache ...")
        for i, first_text in enumerate(sorted(glob.glob(os.path.join(poetry_corpus, '*')))):
            if i % 5 == 0:
                log_it("  We've performed full calculations for %d texts!" % i)
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
        pruned = self._data.copy()
        for count, (one, two) in enumerate(self._data):
            try:
                if count % 1000 == 0:
                    print("We're on entry # %d: that's %d %% done!" % (count, (100 * count/len(self._data))))
                assert os.path.isfile(os.path.join(poetry_corpus, one)), "'%s' does not exist!" % one
                assert os.path.isfile(os.path.join(poetry_corpus, two)), "'%s' does not exist!" % two
                assert one <= two, "%s and %s are mis-ordered!" % (one, two)
                assert self._data[(one, two)].when >= os.path.getmtime(os.path.join(poetry_corpus, one)), "data for '%s' is stale!" % one
                assert self._data[(one, two)].when >= os.path.getmtime(os.path.join(poetry_corpus, two)), "data for '%s' is stale!" % two
                _ = int(self._data[(one, two)].when)
            except (AssertionError, ValueError, KeyError) as err:
                print("Removing entry: (%s, %s)    -- because: %s" % (one, two, err))
                del pruned[(one, two)]
                self._dirty = True
            except BaseException as err:
                print("Unhandled error: %s! Leaving data in place" % err)
        removed = len(self._data) - len(pruned)
        print("Removed %d entries; that's %d %%!" % (removed, 100 * removed/len(self._data)))
        self._data = pruned
        # We're now going to flush the newly cleaned cache directly to disk. Note that
        # we're not UPDATING THE CACHE using flush_cache(), because that would just
        # allow stale old data that we just cleaned out to propagate back in.
        with bz2.open(self._cache_file, 'wb') as pickled_file:
            pickle.dump(self._data, pickled_file, protocol=pickle.HIGHEST_PROTOCOL)
            self._dirty = False
            print("Cache updated!")


class CurrentSimilarityCache(BasicSimilarityCache):
    """A subclass that inherits from whatever class we're currently using for the
    similarity cache without changing any of its behavior. Just a pointer to aid in
    managing this particular issue during development experiments.
    """
    pass


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


force_cache_update = False                  # Set this to True to easily step through this in an IDE.
if force_cache_update:
    print("force_cache_update is True in source; fully populating cache ...")
    build_cache()
    print("update complete, quitting ...")
    sys.exit(0)


if __name__ == "__main__":
    # First: if command-line args are used, just clean, or build, then quit.
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
    import patrick_logger
    patrick_logger.verbosity_level = 3

    c = CurrentSimilarityCache()
    print(c)
    c.clean_cache()
    c.build_cache()

    import random
    for i in range(5000):
        a, b = random.choice(glob.glob(os.path.join(poetry_corpus, '*'))), random.choice(glob.glob(os.path.join(poetry_corpus, '*')))
        print("Similarity between %s and %s is: %.4f" % (os.path.basename(a), os.path.basename(b), c.get_similarity(a,b)))

    c.flush_cache()
