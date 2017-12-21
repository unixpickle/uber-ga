"""
Tests for the random noise APIs.
"""

from functools import partial

import numpy as np

from .noise import NoiseSource

def test_noise_determinism():
    """
    Test that noise from a NoiseSource is deterministic.
    """
    source = NoiseSource(seed=123, size=(1 << 24))
    buf1 = source.block(15, 1).copy()
    buf2 = source.block(15, 2).copy()
    buf3 = source.block(15, 1).copy()
    assert np.allclose(buf1, buf3)
    assert not np.allclose(buf1, buf2)
    source1 = NoiseSource(seed=123, size=(1 << 24))
    buf4 = source1.block(15, 2).copy()
    assert np.allclose(buf4, buf2)
    source1 = NoiseSource(seed=124, size=(1 << 24))
    assert not np.allclose(source1.block(15, 2), source.block(15, 2))

def test_noise_cumulative():
    """
    Test that cumulative_block() gives correct data.
    """
    source = NoiseSource()
    genome = [(1337, 0.5), (123, 0.7), (123123, 1), (3213221, 0.1)]
    expected = np.sum([source.block(15, seed) * scale for seed, scale in genome], axis=0)
    actual = source.cumulative_block(15, genome)
    actual1 = source.cumulative_block(15, genome)
    assert actual.shape == (15,)
    assert np.allclose(actual, actual1)
    assert np.allclose(actual, expected)

def test_noise_block_perf(benchmark):
    """
    Benchmark raw noise generation.
    """
    source = NoiseSource()
    benchmark(partial(source.block, 1000000, 123))

CACHE_TEST_GENOMES = [[],
                      [(1337, 0.5)],
                      [(123, 0.3)],
                      [(1337, 0.5), (555, 0.5)],
                      [(123, 0.3), (444, 1)],
                      [(1337, 0.5), (555, 0.5), (333, 0.2)],
                      [(123, 0.3), (444, 1), (777, 0.8)],
                      [(1337, 0.5), (555, 0.5), (333, 0.2), (912, 0.5)],
                      [(123, 0.3), (444, 1), (777, 0.8), (873, 1)],
                      [(123, 0.3), (444, 1), (777, 0.8), (873, 1), (9823, 1)],
                      [(123, 0.3), (444, 1), (777, 0.8), (873, 1), (9823, 1), (1231, 0.05)]]

def test_noise_cache_perf(benchmark):
    """
    Benchmark successive mutation generation.
    """
    source = NoiseSource()
    def run_ancestors(): # pylint: disable=C0111
        source._cache.clear() # pylint: disable=W0212
        for genome in CACHE_TEST_GENOMES:
            source.cumulative_block(1000000, genome)
    benchmark(run_ancestors)

def test_noise_nocache_perf(benchmark):
    """
    Same as test_noise_cache_perf(), but with no caching.
    """
    source = NoiseSource(max_cache=1)
    def run_ancestors(): # pylint: disable=C0111
        source._cache.clear() # pylint: disable=W0212
        for genome in CACHE_TEST_GENOMES:
            source.cumulative_block(1000000, genome)
    benchmark(run_ancestors)
