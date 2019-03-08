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


import array, bz2, collections, functools, glob, os, pickle, shlex, time

import pid                                              # https://pypi.python.org/pypi/pid/


from utils import *
import poetry_generator as pg

from patrick_logger import log_it       # https://github.com/patrick-brian-mooney/personal-library


@functools.lru_cache(maxsize=8)
def get_mappings(f, markov_length):
    """Trains a generator, then returns the calculated mappings."""
    log_it("get_mappings() called for file %s" % f, 5)
    return pg.PoemGenerator(training_texts=[f], markov_length=markov_length).chains.the_mapping


@functools.lru_cache(maxsize=2048)
def _comparative_form(what):
    """Get the standard form of a text's name for indexing lookup purposes."""
    return os.path.basename(what.strip()).lower()


class BasicSimilarityCache(object):
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

    def __repr__(self):
        try:
            return "< Basic Textual Similarity Cache, with %d results cached >" % len(self._data)
        except AttributeError:
            return "< Basic Textual Similarity Cache (not fully initialized: no data attached) >"
        except BaseException as err:
            return "< Basic Textual Similarity Cache (unknown state because %s) >" % err

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

    @staticmethod
    def calculate_overlap(one, two):
        """Returns the percentage of chains in dictionary ONE that are also in
        dictionary TWO.
        """
        overlap_count = 0
        for which_chain in one.keys():
            if which_chain in two: overlap_count += 1
        return overlap_count / len(one)

    @staticmethod
    def _key_from_texts(one, two):
        """Given texts ONE and TWO, produce a hashable key that can be used to index the
        _data dictionary.
        """
        one, two = _comparative_form(one), _comparative_form(two)
        if one > two:
            one, two = two, one
        return (one, two)

    def _store_data(self, one, two, similarity):
        """Store the SIMILARITY (a float between 0 and 1, weighted toward zero)
        between texts ONE and TWO in the cache.
        """
        key = self._key_from_texts(one, two)
        self._data[key] = {'when': time.time(), 'similarity': similarity, }

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
        self._store_data(one, two, ret)
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
        key = self._key_from_texts(one, two)
        log_it("get_similarity() called for files: %s" % list(key), 5)

        if key in self._data:                       # If it's in the cache, and the data isn't stale ...
            entry = self._data[key]
            if entry['when'] < os.path.getmtime(one):
                log_it("  ... but cached data is stale relative to %s !" % one, 6)
                return self.calculate_similarity(one, two)
            if entry['when'] < os.path.getmtime(two):
                log_it("  ... but cached data is stale relative to %s !" % two, 6)
                return self.calculate_similarity(one, two)
            log_it(" ... returning cached value!", 6)
            return entry['similarity']

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
                if i % 10 == 0:
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


class DeepChainMap(collections.ChainMap):
    """Variant of ChainMap that allows direct updates to inner scopes. Shamelessly
    stolen from the Python documentation for the collections module in the standard
    library, then further adapted.
    """
    def __setitem__(self, key, value):
        for mapping in self.maps:
            if key in mapping:
                mapping[key] = value
                return
        self.maps[0][key] = value

    def __delitem__(self, key):
        for mapping in self.maps:
            if key in mapping:
                del mapping[key]
                return
        raise KeyError(key)


class ChainMapSimilarityCache(BasicSimilarityCache):
    """A similarity cache that keeps its data in multiple shards, in order to limit
    both the file size of individual files and the amount of memory required when
    decompressing.
    """
    _maximum_shard_size = 256 * 1024

    def __init__(self):
        if not os.path.isdir(sharded_cache_location):
            log_it("Directory for storing similarity cache shards does not exist, creating ...")
            os.mkdir(sharded_cache_location)
        shards = list()
        for shard in sorted(glob.glob(os.path.join(sharded_cache_location, '*'))):
            try:
                with bz2.open(shard, mode='rb') as pickled_file:
                    log_it("Loading cached similarity data from shard %s ..." % shlex.quote(shard))
                    shards.append(pickle.load(pickled_file))
            except (OSError, EOFError, AttributeError, pickle.PicklingError) as err:
                log_it("WARNING! Unable to load cached similarity data because %s. Skipping this shard ..." % err)
        self._data = DeepChainMap(*shards)
        self._dirty = False

    def __repr__(self):
        try:
            return "< Fragmented Textual Similarity Cache, in %d shards, with %d results cached >" % (len(self._data.maps), len(self._data))
        except AttributeError:
            return "< Fragmented Textual Similarity Cache (not fully initialized: no data attached) >"
        except BaseException as err:
            return "< Fragmented Textual Similarity Cache (unknown state because %s) >" % err

    def flush_cache(self):
        """Flush the fragmented similarity cache to disk, each shard in a separate
        compressed file. 
        """
        if not self._dirty:
            log_it("Skipping cache update: no changes made!", 3)
            return
        log_it("Updating similarity data cache on disk ...", 1)
        shards_created = set()
        for shard_num, shard in enumerate(self._data.maps):
            try:
                shard_name = os.path.join(sharded_cache_location, "%016d.dat" % shard_num)
                log_it("      ... dumping %d items into shard with name %s" % (len(shard), shlex.quote(shard_name)), 2)
                with bz2.open(shard_name, mode="wb") as pickled_file:
                    pickle.dump(shard, pickled_file, protocol=1)
                shards_created.add(os.path.abspath(shard_name))
            except BaseException as err:
                log_it("WARNING! Unable to dump shard at %s because %s. That data is lost!" % (shlex.quote(shard_name), err))
        extra_shards = [os.path.abspath(i) for i in sorted(glob.glob(os.path.join(sharded_cache_location, "*"))) if os.path.abspath(i) not in shards_created]
        if extra_shards:
            log_it("WARNING: %d leftover shards detected in cache directory! Deleting ..." % len(extra_shards), 1)
            log_it("Extra shards are: %s" % extra_shards, 2)
            for f in sorted(extra_shards):
                log_it("Deleting %s" % shlex.quote(f), 3)
                try:
                    os.unlink(f)
                except BaseException as err:
                    log_it("WARNING! Unable to delete extraneous shard %s!" % f)
        log_it(" ... updated!", 1)
        self._dirty = False

    def _store_data(self, one, two, similarity):
        if len(self._data.maps[0]) >= self._maximum_shard_size:
            self._data = self._data.new_child()
        BasicSimilarityCache._store_data(self, one, two, similarity)


class IndexedArraySimilarityCache(BasicSimilarityCache):
    """Uses Python's array module to store the cached data. There are two arrays:
    one (called _similarity_data) stores similarity scores between texts, and one
    (called _calculation_times) stores the Unix timestamps when the similarity was
    last calculated. A standard Python list (called _index) is used to index the
    arrays.

    This is STILL slower than the BasicSimilarityCache.
    """
    _instance = None

    def __new__(cls, *pargs, **kwargs):
        """Provide basic enforcement of the requirement that this be a singleton class."""
        if not cls._instance:
            cls._instance = BasicSimilarityCache.__new__(cls)
        return cls._instance

    @staticmethod
    def _yielder(value, num_times):
        for i in range(num_times):
            yield value

    def __init__(self, cache_file=similarity_cache_location):
        self._dirty = False
        self._cache_file = cache_file
        try:
            with bz2.open(self._cache_file, mode="rb") as pickled_file:
                self._index = pickle.load(pickled_file)
                self._similarity_data = pickle.load(pickled_file)
                self._calculation_times = pickle.load(pickled_file)
        except BaseException as err:
            log_it("WARNING! Unable to decode similarity cache because %s. Creating new from scratch ..." % err)
            self._index = list()
            self._similarity_data = array.array('f')
            self._calculation_times = array.array('d')

        # Next: if there's not enough space in the underlying arrays, extend them now with empty data that compresses easily.
        num_poems = len(glob.glob(os.path.join(poetry_corpus, '*')))
        if len(self._similarity_data) < (num_poems ** 2) / 2:
            self._similarity_data.extend(self._yielder(-1, ((num_poems ** 2) // 2) - len(self._similarity_data)))
        if len(self._calculation_times) < (num_poems ** 2) / 2:
            self._calculation_times.extend(self._yielder(-1, ((num_poems ** 2) // 2) - len(self._calculation_times)))

    def __repr__(self):
        try:
            return "< IndexedArray Textual Similarity Cache, with %d results cached >" % len(self._index)
        except AttributeError:
            return "< IndexedArray Textual Similarity Cache (not fully initialized: no index!) >"
        except BaseException as err:
            return "< IndexedArray Textual Similarity Cache (unknown state because [ %s ]) >" % err

    @staticmethod
    def _key_name_from_text_names(one, two):
        """Takes ONE and TWO (filenames of source texts) and produces a normalized key
        used to index the similarity cache to find or store the similarity between those
        two texts. Returns a string, which is that key.

        Uses two squiggle arrows pointing in opposite directions as a separator, because
        that currently seems to be highly unlikely to occur in the titles of source
        texts.
        """
        one = os.path.basename(one.strip())
        two = os.path.basename(two.strip())
        if one > two:
            one, two = two, one
        return """%s⇜⇝%s""" % (one, two)

    def flush_cache(self):
        """Flush the cache to disk."""
        if not self._dirty:
            log_it("Skipping cache update: no changes made!", 4)
            return
        log_it("Updating similarity data cache on disk ...", 3)
        with bz2.open(self._cache_file, 'wb') as pickled_file:
            pickle.dump(self._index, pickled_file)
            pickle.dump(self._similarity_data, pickled_file)
            pickle.dump(self._calculation_times, pickled_file)
        log_it(" ... updated successfully!", 3)
        self._dirty = False

    def _index_from_key(self, key):
        """Checks SELF's _index attribute for KEY and returns its numeric index. This
        numeric index is used to find data for KEY in self._similarity_data and
        self._calculation_times. If KEY is not in self._index, returns None.
        """
        try:
            return self._index.index(key)
        except ValueError:
            return None

    def _store_similarity(self, one, two, similarity):
        """Store the calculated SIMILARITY between texts ONE and TWO in the similarity
        cache.
        """
        key = self._key_name_from_text_names(one, two)
        current_index = self._index_from_key(key)
        if not current_index:
            self._index.append(key)
            if len(self._index) > len(self._similarity_data):       # Make room for new data
                self._similarity_data.append(-1)
            if len(self._index) > len(self._calculation_times):     # But only if necessary
                self._calculation_times.append(0)
            current_index = len(self._index) - 1    # reminder: zero-based
        self._similarity_data[current_index] = similarity
        self._calculation_times[current_index] = time.time()
        self._dirty = True

    def calculate_similarity(self, one, two, markov_length=5):
        """Calculate the similarity between texts ONE and TWO, then cache the result."""
        log_it("calculate_similarity() called for: %s" % [one, two], 5)
        if os.path.basename(one.strip()) == os.path.basename(two.strip()):
            return 1                        # Well, that's easy.
        chains_one = get_mappings(one, markov_length)
        chains_two = get_mappings(two, markov_length)
        ret = self.calculate_overlap(chains_one, chains_two) * self.calculate_overlap(chains_two, chains_one)
        self._store_similarity(one, two, ret)
        return ret

    def get_similarity(self, one, two):
        """If the similarity between texts ONE and TWO is cached, and the calculated value
        is not stale, then return it. Otherwise, call calculate_similarity to calculate
        and cache the similarity, then return it.
        """
        key = self._key_name_from_text_names(one, two)
        index_num = self._index_from_key(key)
        if not index_num:
            log_it("  ... not found in cache! Calculating and caching ...", 6)
            return self.calculate_similarity(one, two)
        if self._calculation_times[index_num] < os.path.getmtime(one):
            log_it("  ... but cached data is stale relative to %s !" % one, 6)
            return self.calculate_similarity(one, two)
        if self._calculation_times[index_num] < os.path.getmtime(two):
            log_it("  ... but cached data is stale relative to %s !" % two, 6)
            return self.calculate_similarity(one, two)
        log_it("  ... returning cached value!", 6)
        return self._similarity_data[index_num]

    def clean_cache(self):
        # raise NotImplementedError("#FIXME: cleaning the cache is not yet implemented for the new-style similarity cache!")
        pass


class DictIndexedArraySimilarityCache(IndexedArraySimilarityCache):
    """A subclass of IndexedArraySimilarityCache that maintains its index using a
    dictionary instead of a list. The basic hope is that using a hash map as the
    index will improve indexing performance -- and it works, a little, but it's
    still pretty slow.
    """
    def __init__(self, cache_file=similarity_cache_location):
        self._dirty = False
        self._cache_file = cache_file
        try:
            with bz2.open(self._cache_file, mode="rb") as pickled_file:
                self._index = pickle.load(pickled_file)
                self._similarity_data = pickle.load(pickled_file)
                self._calculation_times = pickle.load(pickled_file)
        except BaseException as err:
            log_it("WARNING! Unable to decode similarity cache because %s. Creating new from scratch ..." % err)
            self._index = dict()
            self._similarity_data = array.array('f')
            self._calculation_times = array.array('d')

        # Next: if there's not enough space in the underlying arrays, extend them now with empty data that compresses easily.
        num_poems = len(glob.glob(os.path.join(poetry_corpus, '*')))
        if len(self._similarity_data) < (num_poems ** 2) / 2:
            self._similarity_data.extend(self._yielder(-1, ((num_poems ** 2) // 2) - len(self._similarity_data)))
        if len(self._calculation_times) < (num_poems ** 2) / 2:
            self._calculation_times.extend(self._yielder(0, ((num_poems ** 2) // 2) - len(self._calculation_times)))

    def __repr__(self):
        try:
            return "< DictIndexedArray Textual Similarity Cache, with %d results cached >" % len(self._index)
        except AttributeError:
            return "< DictIndexedArray Textual Similarity Cache (not fully initialized: no index!) >"
        except BaseException as err:
            return "< DictIndexedArray Textual Similarity Cache (unknown state because [ %s ]) >" % err

    @staticmethod
    def _key_from_text_names(one, two):
        """Takes ONE and TWO (filenames of source texts) and produces a normalized key
        used to index the similarity cache to find or store the similarity between those
        two texts. Returns a tuple, which is that key.
        """
        one = os.path.basename(one.strip())
        two = os.path.basename(two.strip())
        if one > two:
            one, two = two, one
        return (one, two)

    def _index_from_key(self, key):
        """Checks SELF's _index attribute for KEY and returns its numeric index. This
        numeric index is used to find data for KEY in self._similarity_data and
        self._calculation_times. If KEY is not in self._index, returns None.
        """
        try:
            return self._index[key]
        except (KeyError, ):
            return None

    def _store_similarity(self, one, two, similarity):
        """Store the similarity in the cache, lengthening it if necessary.
        """
        key = self._key_name_from_text_names(one, two)
        current_index = self._index_from_key(key)
        if not current_index:
            self._index[key] = 1 + max(self._index.values())
            if len(self._index) > len(self._similarity_data):       # Make room for new data
                self._similarity_data.append(-1)
            if len(self._index) > len(self._calculation_times):     # But only if necessary
                self._calculation_times.append(0)
            current_index = len(self._index) - 1    # reminder: zero-based
        self._similarity_data[current_index] = similarity
        self._calculation_times[current_index] = time.time()
        self._dirty = True


class CurrentSimilarityCache(ChainMapSimilarityCache):
    """A subclass that inherits from whatever class we're currently using for the
    similarity cache without changing any of its behavior. Just a pointer to aid in
    managing this particular issue during development.
    """
    pass


# pandas only used below, in deprecated class. Import it here instead of up top.
import pandas as pd                                     # https://pandas.pydata.org/


class MemoryHogSimilarityCache(BasicSimilarityCache):
    """This BasicSimilarityCache subclass uses a pair of pandas Series to store the
    relevant data. Turns out that pandas dataframes are slow to resize, and multiple
    insertions and data accesses are slow, too, largely (I suspect) because the data
    is not stored efficiently.

    Pickling is also fiddly. Giving up on this.
    """
    _instance = None

    def __new__(cls, *pargs, **kwargs):
        """Enforce the requirement that this be a singleton class"""
        if not cls._instance:
            cls._instance = BasicSimilarityCache.__new__(cls)
        return cls._instance

    def __init__(self, cache_file=similarity_cache_location):
        """Set up a new instance of this object. There should only ever be one.
        Try to read in cached data if it exists in the expected location. If not,
        create a new blank cache.
        """
        self._dirty = False
        self._cache_file = cache_file
        try:
            with bz2.open(self._cache_file, mode='rb') as pickled_file:
                self._similarity_data = pickle.load(pickled_file)
                self._calculation_times = pickle.load(pickled_file)
        except BaseException as err:
            log_it("WARNING! Unable to decode similarity cache because %s. Creating new from scratch ..." % err)
            self._similarity_data = pd.Series(dict(), dtype="float16")
            self._calculation_times = pd.Series(dict(), dtype="float64")

    def __repr__(self):
        try:
            return "< (MemoryHog) Textual Similarity Cache, with %d results cached >" % self._similarity_data.size
        except AttributeError:
            return "< (MemoryHog) Textual Similarity Cache (not fully initialized: no data attached) >"
        except BaseException as err:
            return "< (MemoryHog) Textual Similarity Cache (unknown state because [ %s ]) >" % err

    def clean_cache(self):
        # raise NotImplementedError("#FIXME: cleaning the cache is not yet implemented for the memory-hog-style similarity cache!")
        pass

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
        with bz2.open(self._cache_file, 'wb') as pickled_file:
            pickle.dump(self._similarity_data, pickled_file)
            pickle.dump(self._calculation_times, pickled_file)
        log_it(" ... updated successfully!", 3)
        self._dirty = False

    @staticmethod
    def _key_name_from_text_names(one, two):
        """Takes ONE and TWo (filenames of source texts) and produces a normalized key
        used to index the similarity cache to find or store the similarity between those
        two texts. Returns a string, which is that key.

        Uses two squiggle arrows pointing in opposite directions as a separator, because
        that currently seems to be highly unlikely to occur in the titles of source
        texts.
        """
        one = os.path.basename(one.strip())
        two = os.path.basename(two.strip())
        if one > two:
            one, two = two, one
        return """%s⇜⇝%s""" % (one, two)

    def _store_similarity(self, one, two, similarity):
        """Stores the SIMILARITY (a floating-point number between zero and one, heavily
        weighted toward 0) between ONE and TWO (filenames for texts being compared) in
        the data frames that keep data for this cache. The cache also stores a timestamp
        for when the calculation was performed, which is not passed to this function.
        Instead, the current time is used, on the assumption that the calculation has
        JUST been performed. This could in theory cause unknown stale data in the cache,
        if the source text is updated between when the calculation has been performed
        and the time when the timestamp is finally recorded, but oh well: in this case,
        the gap is generally quite small, in fact, and the implications of stale data
        here are quite small--just that similarity calculations may be using slightly
        incorrect data. Since major updates to source texts are unlikely (we're never
        going to replace the text of 'The Rime of the Ancient Mariner' with the text of
        'The Sick Rose', though we might occasionally correct spelling or punctuation),
        large changes to similarity numbers are also unlikely. We can live with a small,
        unlikely race condition here.

        This function makes no attempt to check whether there is already data stored
        for that key.
        """
        key = self._key_name_from_text_names(one, two)
        self._similarity_data[key] = similarity
        self._calculation_times[key] = time.time()
        self._dirty = True

    def calculate_similarity(self, one, two, markov_length=5):
        log_it("calculate_similarity() called for: %s" % [one, two], 5)
        if one == two:
            return 1                        # Well, that's easy.
        chains_one = get_mappings(one, markov_length)
        chains_two = get_mappings(two, markov_length)
        ret = self.calculate_overlap(chains_one, chains_two) * self.calculate_overlap(chains_two, chains_one)
        self._store_similarity(one, two, ret)
        return ret

    def get_similarity(self, one, two):
        key = self._key_name_from_text_names(one, two)
        if not key in self._similarity_data:
            log_it("  ... not found in cache! Calculating and caching ...", 6)
            return self.calculate_similarity(one, two)
        if self._calculation_times[key] < os.path.getmtime(one):
            log_it("  ... but cached data is stale relative to %s !" % one, 6)
            return self.calculate_similarity(one, two)
        if self._calculation_times[key] < os.path.getmtime(two):
            log_it("  ... but cached data is stale relative to %s !" % two, 6)
            return self.calculate_similarity(one, two)
        log_it("  ... returning cached value!", 6)
        return self._similarity_data[key]


# Numpy is only directly used by the following class.
import numpy as np                                      # http://www.numpy.org/

class VerySlowSimilarityCache(BasicSimilarityCache):
    """This BasicSimilarityCache is a singleton object that manages the global cache of the
    results of similarity calculations. Calculating the similarity between two texts
    is a comparatively time- and memory-intensive operation, so we cache the results
    on disk to speed up textual selection.

    Unlike the previous iteration, this is not implemented under the hood as a
    sorted tuple-indexed dictionary yielding another dictionary: this is a pair
    of pandas DataFrames, so the calculations and cacheing should work much more
    quickly than before. Also unlike its superclass, flushing the cache to disk
    overwrites, rather than

    Interesting object attributes:
      * self._dirty             has the data changed since it was last written to
                                disk?
      * self._cache_file        full path to file on disk where this cache is
                                stored.
      * self._similarity_data   table of zero-weighted two-byte 0-to-1 floats
                                indicating calculated similarity scores. Two bytes
                                is good enough for our purposes here.
      * self._calculation_times table of eight-byte floats encoding Unix timestamp
                                for when calculation was performed. (Eight bytes are
                                needed to keep the precision of the Unix timestamp.)

    This BasicSimilarityCache subclass is VERY VERY SLOW.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        """Enforce the requirement that this be a singleton class"""
        if not cls._instance:
            cls._instance = BasicSimilarityCache.__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self, cache_file=similarity_cache_location):
        """Set up a new instance of this object. There should only ever be one.
        Try to read in cached data if it exists in the expected location. If not,
        create a new blank cache.

        #FIXME: we need to validate the correctness of the data after loading and
        discard any rows or columns whose names are stale! We need to create rows
        and columns for any new texts that have appeared!
        """
        self._dirty = False
        self._cache_file = cache_file
        try:
            with bz2.open(self._cache_file, mode='rb') as pickled_file:
                self._similarity_data = pickle.load(pickled_file)
                self._calculation_times = pickle.load(pickled_file)
        except BaseException as err:
            log_it("WARNING! Unable to decode similarity cache because %s. Creating new from scratch ..." % err)
            poem_files = sorted([os.path.basename(f) for f in glob.glob(os.path.join(poetry_corpus, '*'))])
            self._similarity_data = pd.DataFrame(np.full((len(poem_files), len(poem_files)), np.nan, dtype="float16"), index=poem_files, columns=poem_files)
            self._calculation_times = pd.DataFrame(np.full((len(poem_files), len(poem_files)), np.nan, dtype="float64"), index=poem_files, columns=poem_files)

    def __repr__(self):
        try:
            return "< (BAD-style) Textual Similarity Cache, with %d results cached >" % sum(self._similarity_data.count())
        except AttributeError:
            return "< (BAD-style) Textual Similarity Cache (not fully initialized: no data attached) >"
        except BaseException as err:
            return "< (BAD-style) Textual Similarity Cache (unknown state because %s) >" % err

    def clean_cache(self):
        raise NotImplementedError("#FIXME: cleaning the cache is not yet implemented for the slow-ass similarity cache!")

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
        with bz2.open(self._cache_file, 'wb') as pickled_file:
            pickle.dump(self._similarity_data, pickled_file, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self._calculation_times, pickled_file, protocol=pickle.HIGHEST_PROTOCOL)
        log_it(" ... updated successfully!", 3)
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
        if np.isnan(self._calculation_times[os.path.basename(one)][os.path.basename(two)]):
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
    # Debugging harness, for walking through in a debugger.
    import random
    c = ChainMapSimilarityCache()
    print(c)
    c.build_cache()
    for i in range(5000):
        a, b = random.choice(glob.glob(os.path.join(poetry_corpus, '*'))), random.choice(glob.glob(os.path.join(poetry_corpus, '*')))
        print("Similarity between %s and %s is: %.4f" % (os.path.basename(a), os.path.basename(b), c.get_similarity(a,b)))
    c.flush_cache()
