"""
Virtual batch normalization.
"""

import tensorflow as tf

class VirtualBN:
    """
    A feed-forward network that uses virtual batch norm.
    """
    def __init__(self, epsilon=1e-5):
        self.epsilon = epsilon
        self._unnormed_tensors = []
        self._mean_phs = []
        self._stddev_phs = []
        self._mean_out = []
        self._stddev_out = []

    def add_layer(self, unnormed):
        """
        Add a layer to the network.

        Args:
          unnormed: the unnormalized layer Tensor.

        Returns:
          A Tensor representing the normalized output.
        """
        with tf.variable_scope(None, default_name='virtual_bn'):
            self._unnormed_tensors.append(unnormed)
            depth = unnormed.get_shape()[-1].value
            axes = list(range(len(unnormed.get_shape())-1))
            mean = tf.reduce_mean(unnormed, axis=axes)
            stddev = tf.sqrt(tf.reduce_mean(tf.square(unnormed - mean), axis=axes))
            self._mean_out.append(mean)
            self._stddev_out.append(stddev)
            mean_ph = tf.placeholder(tf.float32, shape=(depth,))
            stddev_ph = tf.placeholder(tf.float32, shape=(depth,))
            self._mean_phs.append(mean_ph)
            self._stddev_phs.append(stddev_ph)
            return (unnormed - mean_ph) / (stddev_ph + self.epsilon)

    def batch_feed(self, sess, feed_dict=None):
        """
        Compute a feed_dict for the reference batch.

        Args:
          sess: the TensorFlow session.
          feed_dict: the initial feed_dict containing the
            reference batch.
        """
        feed_dict = (feed_dict or {}).copy()
        for pairs in zip(self._mean_out, self._stddev_out, self._mean_phs, self._stddev_phs):
            out_pair = sess.run(pairs[:2], feed_dict=feed_dict)
            for placeholder, val in zip(pairs[2:], out_pair):
                feed_dict[placeholder] = val
        return feed_dict
