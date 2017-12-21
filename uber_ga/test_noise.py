"""
Tests for the random noise APIs.
"""

from functools import partial

import numpy as np
import tensorflow as tf

from .noise import NoiseSource, NoiseAdder

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

def test_noise_adder():
    """
    Test that NoiseAdder actually adds/removes noise.
    """
    with tf.Graph().as_default(): # pylint: disable=E1129
        with tf.Session() as sess:
            var1 = tf.Variable(np.array([[1, 2, 3], [4, 5, 6]], dtype='float32'))
            var2 = tf.Variable(np.array([[1, 2], [-1, -2], [3, 3]], dtype='float32'))
            sess.run(tf.global_variables_initializer())
            old_val_1_1, old_val_2_1 = sess.run([var1, var2])

            adder = NoiseAdder(sess, [var1, var2], NoiseSource())
            with adder.seed([(1337, 0.5)]):
                new_val_1_1, new_val_2_1 = sess.run([var1, var2])
            with adder.seed([(1337, 0.1)]):
                new_val_1_2, new_val_2_2 = sess.run([var1, var2])
            with adder.seed([(1337, 0.5)]):
                new_val_1_3, new_val_2_3 = sess.run([var1, var2])
            old_val_1_2, old_val_2_2 = sess.run([var1, var2])
            assert np.allclose(old_val_1_1, old_val_1_2)
            assert np.allclose(old_val_2_1, old_val_2_2)
            assert np.allclose(new_val_1_1, new_val_1_3)
            assert np.allclose(new_val_2_1, new_val_2_3)
            assert not np.allclose(new_val_1_1, new_val_1_2)
            assert not np.allclose(new_val_2_1, new_val_2_2)

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
