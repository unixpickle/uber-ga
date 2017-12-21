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

def test_noise_performance(benchmark):
    """
    Benchmark noise generation.
    """
    source = NoiseSource()
    benchmark(partial(source.block, 1000000, 123))
