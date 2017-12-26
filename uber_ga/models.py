"""
Models suitable for mutation-based training.
"""

from abc import abstractmethod
import math

from anyrl.models import Model
from anyrl.spaces import gym_space_distribution, gym_space_vectorizer

import numpy as np
import tensorflow as tf

def simple_mlp(sess, env, stochastic=False):
    """
    Create a simple MLP policy for the environment.
    """
    return MLP(sess,
               gym_space_distribution(env.action_space),
               gym_space_vectorizer(env.observation_space),
               stochastic,
               (32, 32))

def nature_cnn(sess, env, stochastic=False):
    """
    Create a CNN policy for a game environment.
    """
    return CNN(sess,
               gym_space_distribution(env.action_space),
               gym_space_vectorizer(env.observation_space),
               stochastic)

class FeedforwardPolicy(Model):
    """
    An evolvable feedforward policy.
    """
    def __init__(self, session, action_dist, obs_vectorizer, stochastic):
        """
        Create a policy.

        Args:
          session: the TensorFlow session.
          action_dist: the action Distribution.
          obs_vectorizer: the observation Vectorizer.
          stochastic: if True, sample; otherwise, take the
            mode of the action distribution.
        """
        self.session = session
        self.action_dist = action_dist
        self.obs_vectorizer = obs_vectorizer
        self.stochastic = stochastic
        self.obs_ph = tf.placeholder(tf.float32, shape=(None,) + obs_vectorizer.out_shape)
        output = self.base(int(np.prod(action_dist.param_shape)))
        self.output = tf.reshape(output, (-1,) + action_dist.param_shape)

    @property
    def stateful(self):
        return False

    def start_state(self, batch_size):
        return None

    def step(self, observations, states):
        """
        Apply the model for a single timestep in a batch
        of environments.

        Returns a dict with the following keys:
          'actions': batch of actions, one per env
          'states': new states after the step
          'values': (optional) predicted value function
          'action_params': (optional) parameters that were
            fed into the action distribution.
        """
        feed = {self.obs_ph: self.obs_vectorizer.to_vecs(observations)}
        params = self.session.run(self.output, feed_dict=feed)
        if self.stochastic:
            actions = self.action_dist.sample(params)
        else:
            actions = self.action_dist.mode(params)
        return {'actions': actions, 'states': None}

    @abstractmethod
    def base(self, out_size):
        """
        Produce a flattened action parameter Tensor.
        """
        pass

class MLP(FeedforwardPolicy):
    """
    A multi-layer perceptron model.

    The model is normalized such that the weights can all
    start off with a stddev of 1.
    """
    #pylint: disable=R0913
    def __init__(self, session, action_dist, obs_vectorizer, stochastic, layer_sizes,
                 activation=tf.nn.relu):
        """
        Create an MLP policy.

        Args:
          session: the TF session.
          action_dist: the action distribution.
          obs_vectorizer: the observation vectorizer.
          stochastic: if True, sample actions randomly.
          layer_sizes: a sequence of integers representing
            the number of hidden units in each layer.
          activation: hidden layer activation function.
        """
        self.layer_sizes = layer_sizes
        self.activation = activation
        super(MLP, self).__init__(session, action_dist, obs_vectorizer, stochastic)

    def base(self, out_size):
        in_size = int(np.prod(self.obs_vectorizer.out_shape))
        layer_in = tf.reshape(self.obs_ph, (-1, in_size))
        for size in self.layer_sizes:
            layer_in = tf.layers.dense(layer_in, size,
                                       kernel_initializer=tf.truncated_normal_initializer())
            layer_in = self.activation(tf.contrib.layers.layer_norm(layer_in))
        return tf.layers.dense(layer_in, out_size, kernel_initializer=tf.zeros_initializer())

class CNN(FeedforwardPolicy):
    """
    A CNN model resembling the DQN model.

    The model is normalized such that the weights can all
    start off with a stddev of 1.
    """
    #pylint: disable=R0913
    def __init__(self, session, action_dist, obs_vectorizer, stochastic,
                 activation=tf.nn.relu, input_scale=1/0xff):
        """
        Create an MLP policy.

        Args:
          session: the TF session.
          action_dist: the action distribution.
          obs_vectorizer: the observation vectorizer.
          stochastic: if True, sample actions randomly.
          activation: hidden layer activation function.
        """
        self.activation = activation
        self.input_scale = input_scale
        super(CNN, self).__init__(session, action_dist, obs_vectorizer, stochastic)

    def base(self, out_size):
        conv_kwargs = {
            'activation': self.activation,
            'kernel_initializer': tf.orthogonal_initializer(gain=math.sqrt(2))
        }
        with tf.variable_scope('layer_1'):
            cnn_1 = tf.layers.conv2d(self.obs_ph * self.input_scale, 32, 8, 4, **conv_kwargs)
        with tf.variable_scope('layer_2'):
            cnn_2 = tf.layers.conv2d(cnn_1, 64, 4, 2, **conv_kwargs)
        with tf.variable_scope('layer_3'):
            cnn_3 = tf.layers.conv2d(cnn_2, 64, 3, 1, **conv_kwargs)
        flat_size = np.prod(cnn_3.get_shape()[1:])
        flat_in = tf.reshape(cnn_3, (tf.shape(cnn_3)[0], int(flat_size)))
        with tf.variable_scope('hidden'):
            hidden = tf.layers.dense(flat_in, 512, **conv_kwargs)
        with tf.variable_scope('output'):
            return tf.layers.dense(hidden, out_size, kernel_initializer=tf.zeros_initializer())
