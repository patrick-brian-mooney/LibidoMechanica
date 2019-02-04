#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""similarity_cache.py is a utility class for Patrick Mooney's LibidoMechanica
project. It provides a class that tracks calculated "similarity" between texts
in the poetry corpus. This is an expensive calculation whose value needs to be
known repeatedly over multiple runs of the script, so it's saved once made.

This file is part of the LibidoMechanica scripts, a project that is copyright
2016-19 by Patrick Mooney. It is alpha software and the author releases it
ABSOLUTELY WITHOUT WARRANTY OF ANY KIND. You are welcome to use it under the
terms of the GNU General Public License, either version 3 or (at your option)
any later version. See the file LICENSE.md for details.
"""


import bz2, functools, glob, os, pickle, time

import pid                                              # https://pypi.python.org/pypi/pid/
import pandas as pd                                     # https://pandas.pydata.org/
import numpy as np                                      # http://www.numpy.org/


from utils import *
import poetry_generator as pg

from patrick_logger import log_it       # https://github.com/patrick-brian-mooney/personal-library


@functools.lru_cache(maxsize=8)
def get_mappings(f, markov_length):
    """Trains a generator, then returns the calculated mappings."""
    log_it("get_mappings() called for file %s" % f, 5)
    return pg.PoemGenerator(training_texts=[f], markov_length=markov_length).chains.the_mapping


class OldSimilarityCache(object):
    """This class is an object that manages the global cache of text similarities.

    The object's internal data cache is a dictionary:
        { (text_name_one, text_name_two):         (a tuple)
              { 'when':,                          (a datetime: when the calculation was made)
                'similarity':                     (a value between 0 and 1, rather heavily weighted toward zero)
                  }
             }
    """
    def __init__(self, cache_file=similarity_cache_location):
        try:
            with bz2.open(similarity_cache_location, "rb") as pickled_file:
                log_it("Loading cached similarity data ...", 3)
                self._data = pickle.load(pickled_file)
        except (OSError, EOFError, AttributeError, pickle.PicklingError) as err:
            log_it("WARNING! Unable to load cached similarity data because %s. Preparing empty cache ..." % err)
            self._data = dict()
        self._dirty = False
        self._cache_file = cache_file

    def __str__(self):
        try:
            return "< Textual Similarity Cache, with %d results cached >" % len(self._data)
        except AttributeError:
            return "< Textual Similarity Cache (not fully initialized: no data attached) >"
        except BaseException as err:
            return "< Textual Similarity Cache (unknown state because %s) >" % err

    def flush_cache(self):
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
        overhead of keeping all this data in memory could in theory grow quite large.
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
            log_it("Unhandled exception occurred!   %s" % err)
        with bz2.open(self._cache_file, 'wb') as pickled_file:
            pickle.dump(self._data, pickled_file, protocol=pickle.HIGHEST_PROTOCOL)
        log_it(" ... updated!", 3)
        self._dirty = False

    @staticmethod
    def calculate_overlap(one, two):
        """return the percentage of chains in dictionary ONE that are also in
        dictionary TWO.
        """
        overlap_count = 0
        for which_chain in one.keys():
            if which_chain in two: overlap_count += 1
        return overlap_count / len(one)

    def calculate_similarity(self, one, two, markov_length=5):
        """Come up with a score evaluating how similar the two texts are to each other.
        This actually means, more specifically, "the product of (a) the percentage of
        chains in the set of chains of length MARKOV_LENGTH constructed from text ONE
        that are also in text TWO; multiplied by (b) the percentage of chains of
        length MARKOV_LENGTH constructed from text TWO that are also in chains
        constructed from text ONE.

        This routine also caches the calculated result in the global similarity cache.
        It's a comparatively expensive calculation to make, so we store the results.
        """
        log_it("calculate_similarity() called for: %s" % [one, two], 5)
        if one == two:
            return 1                        # Well, that's easy.
        chains_one = get_mappings(one, markov_length)
        chains_two = get_mappings(two, markov_length)
        ret = self.calculate_overlap(chains_one, chains_two) * self.calculate_overlap(chains_two, chains_one)
        self._data[tuple(sorted([os.path.basename(one), os.path.basename(two)]))] = {'when': time.time(), 'similarity': ret,}
        self._dirty = True
        return ret

    def get_similarity(self, one, two):
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
        index = tuple(sorted([os.path.basename(one), os.path.basename(two)]))
        log_it("get_similarity() called for files: %s" % list(index), 5)

        if index in self._data:                       # If it's in the cache, and the data isn't stale ...
            if self._data[index]['when'] < os.path.getmtime(one):
                log_it("  ... but cached data is stale relative to %s !" % one, 6)
                return self.calculate_similarity(one, two)
            if self._data[index]['when'] < os.path.getmtime(two):
                log_it("  ... but cached data is stale relative to %s !" % two, 6)
                return self.calculate_similarity(one, two)
            log_it(" ... returning cached value!", 6)
            return self._data[index]['similarity']

        log_it(" ... not found in cache! Calculating and cacheing ...", 6)
        return self.calculate_similarity(one, two)

    def build_cache(self):
        """Sequentially go through the corpus, text by text, forcing comparisons to all
        other texts and cacheing the results, to make sure the cache is fully
        populated. Periodically, it dumps the results to disk by updating the on-disk
        cache, so that not all of the calculation results are lost if the run is
        interrupted. Note that "saves" are actually updates, because it's possible that
        there are multiple instances of this script running -- say, one that is
        intentionally running this function in an IDE, and another that's run
        automatically as a CRON job to generate a poem, but which also winds up
        calculating new similarity data that it also saves. Concurrent runs do not
        share the in-memory copy of the cache, and the second ("generating") run
        mentioned above may wind up duplicating the work of the first. Hence "update,"
        above. The worst-case scenario here is a corrupted cache, which SHOULD be newly
        created by script runs. Hopefully. In any case, if not, hey, it's just a cache.

        This method takes a VERY long time to run if starting from an empty cache with
        many source texts in the corpus. The cache CAN OF COURSE be allowed to
        populate itself across multiple runs.
        """
        log_it("Building cache ...")
        for i, first_text in enumerate(sorted(glob.glob(os.path.join(poetry_corpus, '*')))):
            if i % 10 == 0:
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
        of the data, then rebind the copy over the original data after cleaning is
        done.
        """
        pruned = self._data.copy()
        for count, (one, two) in enumerate(self._data):
            try:
                if count % 1000 == 0:
                    log_it("We're on entry # %d: that's %d %% done!" % (count, (100 * count/len(self._data))))
                assert os.path.isfile(os.path.join(poetry_corpus, one)), "'%s' does not exist!" % one
                assert os.path.isfile(os.path.join(poetry_corpus, two)), "'%s' does not exist!" % two
                assert one <= two, "%s and %s are mis-ordered!" % (one, two)
                assert self._data[(one, two)]['when'] >= os.path.getmtime(os.path.join(poetry_corpus, one)), "data for '%s' is stale!" % one
                assert self._data[(one, two)]['when'] >= os.path.getmtime(os.path.join(poetry_corpus, two)), "data for '%s' is stale!" % two
                _ = int(self._data[(one, two)]['when'])
            except (AssertionError, ValueError, KeyError) as err:
                log_it("Removing entry: (%s, %s)    -- because: %s" % (one, two, err))
                del pruned[(one, two)]
                self._dirty = True
            except BaseException as err:
                log_it("Unhandled error: %s! Leaving data in place" % err)
        removed = len(self._data) - len(pruned)
        log_it("Removed %d entries; that's %d %%!" % (removed, 100 * removed/len(self._data)))
        self._data = pruned
        # We're now going to flush the newly cleaned cache directly to disk. Note that
        # we're not UPDATING THE CACHE using flush_cache(), because that would just
        # allow stale old data that we just cleaned out to propagate back in.
        with bz2.open(self._cache_file, 'wb') as pickled_file:
            pickle.dump(self._data, pickled_file, protocol=pickle.HIGHEST_PROTOCOL)
            self._dirty = False
            log_it("Cache updated!")


class SimilarityCache(OldSimilarityCache):
    pass


class NewSimilarityCache(SimilarityCache):
    def __init__(self, cache_file=similarity_cache_location):
        poem_files = sorted([os.path.basename(f) for f in glob.glob(os.path.join(poetry_corpus, '*'))])
        self._similarity_data = pd.DataFrame(np.full((len(poem_files), len(poem_files)), -1, dtype="float16"), index=poem_files, columns=poem_files)
        self._calculation_times = pd.DataFrame(np.full((len(poem_files), len(poem_files)), -1, dtype="float64"), index=poem_files, columns=poem_files)
        #FIXME: now load cached data, if available
        self._dirty = False
        self._cache_file = cache_file

    def __str__(self):                                          #FIXME
        try:
            return "< (new-style) Textual Similarity Cache, with %d results cached >" % len(self._data)
        except AttributeError:
            return "< (new-style) Textual Similarity Cache (not fully initialized: no data attached) >"
        except BaseException as err:
            return "< (new-style) Textual Similarity Cache (unknown state because %s) >" % err

    def clean_cache(self):
        raise NotImplementedError("#FIXME: cleaning the cache is not yet implemented for the new-style similarity cache!")

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
        """
        if not self._dirty:
            log_it("Skipping cache update: no changes made!", 4)
            return
        log_it("Updating similarity data cache on disk ...", 3)
        with bz2.open(self._cache_file + '.new', 'wb') as pickled_file:
            pickle.dump(self._similarity_data, pickled_file, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self._calculation_times, pickled_file, protocol=pickle.HIGHEST_PROTOCOL)
        log_it(" ... updated!", 3)
        self._dirty = False

    def calculate_similarity(self, one, two, markov_length=5):
        """Come up with a score evaluating how similar the two texts are to each other.
        This actually means, more specifically, "the product of (a) the percentage of
        chains in the set of chains of length MARKOV_LENGTH constructed from text ONE
        that are also in text TWO; multiplied by (b) the percentage of chains of
        length MARKOV_LENGTH constructed from text TWO that are also in chains
        constructed from text ONE.

        This routine also caches the calculated result in the global similarity cache.
        It's a comparatively expensive calculation to make, so we store the results.
        """
        log_it("calculate_similarity() called for: %s" % [one, two], 5)
        if one == two:
            return 1                        # Well, that's easy.
        elif one > two:
            one, two = two, one
        chains_one = get_mappings(one, markov_length)
        chains_two = get_mappings(two, markov_length)
        ret = self.calculate_overlap(chains_one, chains_two) * self.calculate_overlap(chains_two, chains_one)
        one, two = os.path.basename(one), os.path.basename(two)
        self._similarity_data[one][two] = ret
        self._calculation_times[one][two] = time.time()
        self._dirty = True
        return ret

    def get_similarity(self, one, two):
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
        log_it("get_similarity() called for files: %s" % [one, two], 5)
        if one > two:
            one, two = two, one
        if self._calculation_times[os.path.basename(one)][os.path.basename(two)] < 0:
            log_it("  ... not found in cache! Calculating and caching ...", 6)
            return self.calculate_similarity(one, two)
        if self._calculation_times[os.path.basename(one)][os.path.basename(two)] < os.path.getmtime(one):
            log_it("  ... but cached data is stale relative to %s !" % one, 6)
            return self.calculate_similarity(one, two)
        if self._calculation_times[os.path.basename(one)][os.path.basename(two)] < os.path.getmtime(two):
            log_it("  ... but cached data is stale relative to %s !" % two, 6)
            return self.calculate_similarity(one, two)
        log_it("  ... returning cached value!", 6)
        return self._similarity_data[os.path.basename(one)][os.path.basename(two)]



if __name__ == "__main__":
    # Unit tests
    c = NewSimilarityCache()
    print(c)
    c.build_cache()
