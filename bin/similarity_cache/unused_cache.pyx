#!/usr/bin/python3.5
"""This module holds old implementations of the similarity cache. None of them
are currently used.

This script is part of LibidoMechanica, a blog of automatically generated text.
LibidoMechanica is copyright 2016-20 by Patrick Mooney. It is licensed under
the GPC, either version 3 or (at your option) any later version. See the file
LICENSE.md for details.
"""


import array
import bz2
import glob
import os
from pathlib import Path
import pickle
import time
import typing


from bin.globs import *
from bin.globs import similarity_cache_location, poetry_corpus
from bin.similarity_cache import similarity_cache as sc


import pandas as pd                                     # https://pandas.pydata.org/
import numpy as np                                      # http://www.numpy.org/


class MemoryHogSimilarityCache(sc.BasicSimilarityCache):
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
            cls._instance = sc.BasicSimilarityCache.__new__(cls)
        return cls._instance

    def __init__(self, cache_file: typing.Union[str, Path]=similarity_cache_location):
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
            print("WARNING! Unable to decode similarity cache because %s. Creating new from scratch ..." % err)
            self._similarity_data = pd.Series(dict(), dtype="float16")
            self._calculation_times = pd.Series(dict(), dtype="float64")

    def __repr__(self) -> str:
        try:
            return "< (MemoryHog) Textual Similarity Cache, with %d results cached >" % self._similarity_data.size
        except AttributeError:
            return "< (MemoryHog) Textual Similarity Cache (not fully initialized: no data attached) >"
        except BaseException as err:
            return "< (MemoryHog) Textual Similarity Cache (unknown state because [ %s ]) >" % err

    def clean_cache(self):
        raise NotImplementedError("#FIXME: cleaning the cache is not yet implemented for the memory-hog-style similarity cache!")

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
            sc.log_it("Skipping cache update: no changes made!", 4)
            return
        sc.log_it("Updating similarity data cache on disk ...", 3)
        with bz2.open(self._cache_file, 'wb') as pickled_file:
            pickle.dump(self._similarity_data, pickled_file)
            pickle.dump(self._calculation_times, pickled_file)
        sc.log_it(" ... updated successfully!", 3)
        self._dirty = False

    @staticmethod
    def _key_name_from_text_names(one: str, two: str) -> str:
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

    def _store_similarity(self, one: typing.Union[str, Path],
                          two: typing.Union[str, Path],
                          similarity: float):
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

    def calculate_similarity(self, one: typing.Union[str, Path],
                             two: typing.Union[str, Path],
                             markov_length: int=5) -> float:
        sc.log_it("calculate_similarity() called for: %s" % [one, two], 5)
        if one == two:
            return 1                        # Well, that's easy.
        chains_one = sc.get_mappings(one, markov_length)
        chains_two = sc.get_mappings(two, markov_length)
        ret = self.calculate_overlap(chains_one, chains_two) * self.calculate_overlap(chains_two, chains_one)
        self._store_similarity(one, two, ret)
        return ret

    def get_similarity(self, one: typing.Union[str, Path],
                       two: typing.Union[str, Path]) -> float:
        key = self._key_name_from_text_names(one, two)
        if not key in self._similarity_data:
            sc.log_it("  ... not found in cache! Calculating and caching ...", 6)
            return self.calculate_similarity(one, two)
        if self._calculation_times[key] < os.path.getmtime(one):
            sc.log_it("  ... but cached data is stale relative to %s !" % one, 6)
            return self.calculate_similarity(one, two)
        if self._calculation_times[key] < os.path.getmtime(two):
            sc.log_it("  ... but cached data is stale relative to %s !" % two, 6)
            return self.calculate_similarity(one, two)
        sc.log_it("  ... returning cached value!", 6)
        return self._similarity_data[key]


class ShardedChainMap(collections.ChainMap):
    """Variant of ChainMap that allows direct updates to inner scopes. Shamelessly
    stolen from the Python documentation for the collections module in the standard
    library, then further adapted.
    """
    _maximum_shard_size = 2 ** 16

    def __repr__(self) -> str:
        try:
            return "< ShardedChainMap, in %d shards, with %d results cached >" % (len(self.maps), len(self))
        except BaseException as err:
            return "< ShardedChainMap (unknown state because %s) >" % err

    def __setitem__(self,
                    key: typing.Tuple[typing.Union[str, Path], typing.Union[str, Path]],
                    data):
        for mapping in self.maps:                           # If it's already in one of the underlying dicts, update it
            if key in mapping:
                mapping[key] = data
                return
        for mapping in self.maps:
            if len(mapping) < self._maximum_shard_size:    # Otherwise, find a non-full shard to add it to if possible
                mapping[key] = data
                return
        self.maps = [dict()] + self.maps                    # Otherwise, create a new shard and use it to store the value
        self.maps[0][key] = data

    def __delitem__(self,
                    key: typing.Tuple[typing.Union[str, Path], typing.Union[str, Path]]):
        for mapping in self.maps:
            if key in mapping:
                del mapping[key]
                return
        raise KeyError(key)


cdef class ChainMapSimilarityCache(sc.BasicSimilarityCache):
    """A similarity cache that keeps its data in multiple shards, in order to limit
    both the file size of individual files and the amount of memory required when
    decompressing.
    """
    def __init__(self):                                     #cpdef
        if not os.path.isdir(sharded_cache_location):
            print("Directory for storing similarity cache shards does not exist, creating ...")
            os.mkdir(sharded_cache_location)
        shards = list()
        for shard in sorted(glob.glob(os.path.join(sharded_cache_location, '*'))):
            try:
                with bz2.open(shard, mode='rb') as pickled_file:
                    print("Loading cached similarity data from shard %s ..." % shlex.quote(shard))
                    shards.append(pickle.load(pickled_file))
            except (OSError, EOFError, AttributeError, pickle.PicklingError) as err:
                print("WARNING! Unable to load cached similarity data because %s. Skipping this shard ..." % err)
        self._data = ShardedChainMap(*shards)
        self._dirty = False

    def __repr__(self) -> str:
        try:
            return "< Fragmented Textual Similarity Cache, in %d shards, with %d results cached >" % (len(self._data.maps), len(self._data))
        except AttributeError:
            return "< Fragmented Textual Similarity Cache (not fully initialized: no data attached) >"
        except BaseException as err:
            return "< Fragmented Textual Similarity Cache (unknown state because %s) >" % err

    cpdef flush_cache(self):
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
                    log_it("WARNING! Unable to delete extraneous shard %s because %s!" % (f, err))
        log_it(" ... updated!", 1)
        self._dirty = False

    cpdef clean_cache(self):
        """Clean out stale and malformed data from the cache, then write it back to disk.
        Goes the entire sharded dictionary, key by key, building a new dictionary of
        validated items, then swaps that in and flushes the data to disk.
        """
        current_texts = _all_poems_in_comparative_form()
        entries_validated = 0
        pruned = ShardedChainMap()
        for key in self._data:
            try:
                assert key not in pruned, "Duplicate key [ %s ] found! ... cleaning." % (key, )
                one, two = key
                assert one <= two, "texts in key %s are mis-ordered!" % list(key)
                assert one in current_texts, "%s no longer exists on disk!" % one
                assert two in current_texts, "%s no longer exists on disk!" % two
                assert self._data[key].when >= os.path.getmtime(current_texts[one]), "data for '%s' is stale!" % current_texts[one]
                assert self._data[key].when >= os.path.getmtime(current_texts[two]), "data for '%s' is stale!" % current_texts[two]
                _ = float(self._data[key].when)
                # If we made it this far ...
                pruned[key] = self._data[key]
            except (AssertionError, ValueError, KeyError, OSError) as err:
                print("Removing entry: [ %s ]    -- because: %s" % (key, err))
                self._dirty = True
            entries_validated += 1
            if entries_validated % 1000 == 0:
                print("We've validated %d entries, that's %.04f%%!" % (entries_validated, 100*(entries_validated/len(self._data))))
        entries_cleaned = len(self._data) - len(pruned)
        print("DONE! Eliminated %d stale entries (that's %.04f%%)." % (entries_cleaned, 100 * (entries_cleaned/len(self._data))))
        self._data = pruned
        self.flush_cache()

class VerySlowSimilarityCache(sc.BasicSimilarityCache):
    """This BasicSimilarityCache is a singleton object that manages the global cache of the
    results of similarity calculations. Calculating the similarity between two texts
    is a comparatively time- and memory-intensive operation, so we cache the results
    on disk to speed up textual selection.

    Unlike the previous iteration, this is not implemented under the hood as a
    sorted tuple-indexed dictionary yielding another dictionary: this is a pair
    of pandas DataFrames, so the calculations and cacheing should work much more
    quickly than before. (EDIT. One would think. This is still quite slow.) Also
    unlike its superclass, flushing the cache to disk overwrites, rather than
    updates, the data.

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
            cls._instance = sc.BasicSimilarityCache.__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self, cache_file: typing.Union[str, Path]=similarity_cache_location):
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
            print("WARNING! Unable to decode similarity cache because %s. Creating new from scratch ..." % err)
            poem_files = sorted([os.path.basename(f) for f in glob.glob(os.path.join(poetry_corpus, '*'))])
            self._similarity_data = pd.DataFrame(np.full((len(poem_files), len(poem_files)), np.nan, dtype="float16"), index=poem_files, columns=poem_files)
            self._calculation_times = pd.DataFrame(np.full((len(poem_files), len(poem_files)), np.nan, dtype="float64"), index=poem_files, columns=poem_files)

    def __repr__(self) -> str:
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
            sc.log_it("Skipping cache update: no changes made!", 4)
            return
        sc.log_it("Updating similarity data cache on disk ...", 3)
        with bz2.open(self._cache_file, 'wb') as pickled_file:
            pickle.dump(self._similarity_data, pickled_file, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self._calculation_times, pickled_file, protocol=pickle.HIGHEST_PROTOCOL)
        sc.log_it(" ... updated successfully!", 3)
        self._dirty = False

    def calculate_similarity(self, one: typing.Union[str, Path],
                             two: typing.Union[str, Path],
                             markov_length: int=5) -> float:
        """Come up with a score evaluating how similar the two texts are to each other.
        This actually means, more specifically, "the product of (a) the percentage of
        chains in the set of chains of length MARKOV_LENGTH constructed from text ONE
        that are also in text TWO; multiplied by (b) the percentage of chains of
        length MARKOV_LENGTH constructed from text TWO that are also in chains
        constructed from text ONE.

        This routine also caches the calculated result in the global similarity cache.
        It's a comparatively expensive calculation to make, so we store the results.
        """
        sc.log_it("calculate_similarity() called for: %s" % [one, two], 5)
        if one == two:
            return 1                        # Well, that's easy.
        elif one > two:
            one, two = two, one
        chains_one = sc.get_mappings(one, markov_length)
        chains_two = sc.get_mappings(two, markov_length)
        ret = self.calculate_overlap(chains_one, chains_two) * self.calculate_overlap(chains_two, chains_one)
        one, two = os.path.basename(one), os.path.basename(two)
        self._similarity_data[one][two] = ret
        self._calculation_times[one][two] = time.time()
        self._dirty = True
        return ret

    def get_similarity(self, one: typing.Union[str, Path],
                       two: typing.Union[str, Path]) -> float:
        """Checks to see if the similarity between ONE and TWO is already known. If it is,
        returns that similarity from the cache. Otherwise, calculates the similarity and
        stores it in the global similarity cache, which is written at the end of the
        script's run.

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
        sc.log_it("get_similarity() called for files: %s" % [one, two], 5)
        if one > two:
            one, two = two, one
        if np.isnan(self._calculation_times[os.path.basename(one)][os.path.basename(two)]):
            sc.log_it("  ... not found in cache! Calculating and caching ...", 6)
            return self.calculate_similarity(one, two)
        if self._calculation_times[os.path.basename(one)][os.path.basename(two)] < os.path.getmtime(one):
            sc.log_it("  ... but cached data is stale relative to %s !" % one, 6)
            return self.calculate_similarity(one, two)
        if self._calculation_times[os.path.basename(one)][os.path.basename(two)] < os.path.getmtime(two):
            sc.log_it("  ... but cached data is stale relative to %s !" % two, 6)
            return self.calculate_similarity(one, two)
        sc.log_it("  ... returning cached value!", 6)
        return self._similarity_data[os.path.basename(one)][os.path.basename(two)]


class IndexedArraySimilarityCache(sc.BasicSimilarityCache):
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
            cls._instance = sc.BasicSimilarityCache.__new__(cls)
        return cls._instance

    @staticmethod
    def _yielder(value: int, num_times: int) -> int:
        for i in range(num_times):
            yield value

    def __init__(self, cache_file: typing.Union[str, Path]=similarity_cache_location):
        self._dirty = False
        self._cache_file = cache_file
        try:
            with bz2.open(self._cache_file, mode="rb") as pickled_file:
                self._index = pickle.load(pickled_file)
                self._similarity_data = pickle.load(pickled_file)
                self._calculation_times = pickle.load(pickled_file)
        except BaseException as err:
            print("WARNING! Unable to decode similarity cache because %s. Creating new from scratch ..." % err)
            self._index = list()
            self._similarity_data = array.array('f')
            self._calculation_times = array.array('d')

        # Next: if there's not enough space in the underlying arrays, extend them now with empty data that compresses easily.
        num_poems = len(glob.glob(os.path.join(poetry_corpus, '*')))
        if len(self._similarity_data) < (num_poems ** 2) / 2:
            self._similarity_data.extend(self._yielder(-1, ((num_poems ** 2) // 2) - len(self._similarity_data)))
        if len(self._calculation_times) < (num_poems ** 2) / 2:
            self._calculation_times.extend(self._yielder(-1, ((num_poems ** 2) // 2) - len(self._calculation_times)))

    def __repr__(self) -> None:
        try:
            return "< IndexedArray Textual Similarity Cache, with %d results cached >" % len(self._index)
        except AttributeError:
            return "< IndexedArray Textual Similarity Cache (not fully initialized: no index!) >"
        except BaseException as err:
            return "< IndexedArray Textual Similarity Cache (unknown state because [ %s ]) >" % err

    @staticmethod
    def _key_name_from_text_names(one: str, two: str) -> str:
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
            sc.log_it("Skipping cache update: no changes made!", 4)
            return
        sc.log_it("Updating similarity data cache on disk ...", 3)
        with bz2.open(self._cache_file, 'wb') as pickled_file:
            pickle.dump(self._index, pickled_file)
            pickle.dump(self._similarity_data, pickled_file)
            pickle.dump(self._calculation_times, pickled_file)
        sc.log_it(" ... updated successfully!", 3)
        self._dirty = False

    def _index_from_key(self, key) -> int:
        """Checks SELF's _index attribute for KEY and returns its numeric index. This
        numeric index is used to find data for KEY in self._similarity_data and
        self._calculation_times. If KEY is not in self._index, returns None.
        """
        try:
            return self._index.index(key)
        except ValueError:
            return None

    def _store_similarity(self, one: typing.Union[str, Path],
                          two: typing.Union[str, Path],
                          similarity: float):
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

    def calculate_similarity(self, one: typing.Union[str, Path],
                             two: typing.Union[str, Path],
                             markov_length: int=5) -> float:
        """Calculate the similarity between texts ONE and TWO, then cache the result."""
        sc.log_it("calculate_similarity() called for: %s" % [one, two], 5)
        if os.path.basename(one.strip()) == os.path.basename(two.strip()):
            return 1                        # Well, that's easy.
        chains_one = sc.get_mappings(one, markov_length)
        chains_two = sc.get_mappings(two, markov_length)
        ret = self.calculate_overlap(chains_one, chains_two) * self.calculate_overlap(chains_two, chains_one)
        self._store_similarity(one, two, ret)
        return ret

    def get_similarity(self, one: typing.Union[str, Path],
                       two: typing.Union[str, Path]) -> float:
        """If the similarity between texts ONE and TWO is cached, and the calculated value
        is not stale, then return it. Otherwise, call calculate_similarity to calculate
        and cache the similarity, then return it.
        """
        key = self._key_name_from_text_names(one, two)
        index_num = self._index_from_key(key)
        if not index_num:
            sc.log_it("  ... not found in cache! Calculating and caching ...", 6)
            return self.calculate_similarity(one, two)
        if self._calculation_times[index_num] < os.path.getmtime(one):
            sc.log_it("  ... but cached data is stale relative to %s !" % one, 6)
            return self.calculate_similarity(one, two)
        if self._calculation_times[index_num] < os.path.getmtime(two):
            sc.log_it("  ... but cached data is stale relative to %s !" % two, 6)
            return self.calculate_similarity(one, two)
        sc.log_it("  ... returning cached value!", 6)
        return self._similarity_data[index_num]

    def clean_cache(self):
        raise NotImplementedError("#FIXME: cleaning the cache is not yet implemented for the indexed-array similarity cache!")


class DictIndexedArraySimilarityCache(IndexedArraySimilarityCache):
    """A subclass of IndexedArraySimilarityCache that maintains its index using a
    dictionary instead of a list. The basic hope is that using a hash map as the
    index will improve indexing performance -- and it works, a little, but it's
    still pretty slow.
    """
    def __init__(self, cache_file: typing.Union[str, Path]=similarity_cache_location):
        self._dirty = False
        self._cache_file = cache_file
        try:
            with bz2.open(self._cache_file, mode="rb") as pickled_file:
                self._index = pickle.load(pickled_file)
                self._similarity_data = pickle.load(pickled_file)
                self._calculation_times = pickle.load(pickled_file)
        except BaseException as err:
            print("WARNING! Unable to decode similarity cache because %s. Creating new from scratch ..." % err)
            self._index = dict()
            self._similarity_data = array.array('f')
            self._calculation_times = array.array('d')

        # Next: if there's not enough space in the underlying arrays, extend them now with empty data that compresses easily.
        num_poems = len(glob.glob(os.path.join(poetry_corpus, '*')))
        if len(self._similarity_data) < (num_poems ** 2) / 2:
            self._similarity_data.extend(self._yielder(-1, ((num_poems ** 2) // 2) - len(self._similarity_data)))
        if len(self._calculation_times) < (num_poems ** 2) / 2:
            self._calculation_times.extend(self._yielder(0, ((num_poems ** 2) // 2) - len(self._calculation_times)))

    def __repr__(self) -> str:
        try:
            return "< DictIndexedArray Textual Similarity Cache, with %d results cached >" % len(self._index)
        except AttributeError:
            return "< DictIndexedArray Textual Similarity Cache (not fully initialized: no index!) >"
        except BaseException as err:
            return "< DictIndexedArray Textual Similarity Cache (unknown state because [ %s ]) >" % err

    @staticmethod
    def _key_from_text_names(one: str, two: str) -> typing.Tuple[str, str]:
        """Takes ONE and TWO (filenames of source texts) and produces a normalized key
        used to index the similarity cache to find or store the similarity between those
        two texts. Returns a tuple, which is that key.
        """
        one = os.path.basename(one.strip())
        two = os.path.basename(two.strip())
        if one > two:
            one, two = two, one
        return (one, two)

    def _index_from_key(self, key) -> int:
        """Checks SELF's _index attribute for KEY and returns its numeric index. This
        numeric index is used to find data for KEY in self._similarity_data and
        self._calculation_times. If KEY is not in self._index, returns None.
        """
        try:
            return self._index[key]
        except (KeyError, ):
            return None

    def _store_similarity(self, one: typing.Union[str, Path],
                          two: typing.Union[str, Path],
                          similarity: float):
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
