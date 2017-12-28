"""
Genetic algorithm outer loop.
"""

# Avoid MPI errors:
# pylint: disable=E1101

import random

from mpi4py import MPI
import tensorflow as tf

from .noise import NoiseSource, NoiseAdder, noise_seed

class LearningSession:
    """
    A GA optimization session.
    """
    def __init__(self, session, model, variables=None, noise=None):
        self.session = session
        self.model = model
        self.variables = (variables or tf.trainable_variables())
        self.noise = (noise or NoiseSource())
        self.parents = [()]
        self._noise_adder = NoiseAdder(self.session, self.variables, self.noise)
        _synchronize_variables(self.session, self.variables)

    def export_state(self):
        """
        Export the state of the learning session to a
        picklable object.

        This does not include the TensorFlow graph itself,
        but it does include the initialization.
        """
        return {
            'variables': self.session.run(self.variables),
            'parents': self.parents,
            'noise': self.noise
        }

    def import_state(self, state):
        """
        Import a state exported by export_state().

        This assumes that the LearningSession has already
        been setup with a suitable TensorFlow graph.

        This may add nodes (e.g. assigns) to the graph, so
        it should not be called often.
        """
        for var, val in zip(self.variables, state['variables']):
            self.session.run(tf.assign(var, val))
        self.parents = state['parents']
        self._noise_adder.noise = state['noise']

    # pylint: disable=R0913
    def generation(self, env, trials=1, truncation=10, population=5000, stddev=0.1):
        """
        Run a generation of the algorithm and update
        self.parents for the new generation.

        Call this from each MPI worker.

        Args:
          env: the gym.Env to use to evaluate the model.
          trials: the number of episodes to run.
          truncation: the number of parents to keep.
          population: the number of genomes to try.
          stddev: mutation standard deviation.

        Returns a sorted list of (rew, genome) tuples.
        Updates self.parents to reflect the elite.
        """
        res = []
        for i in range(MPI.COMM_WORLD.Get_rank(), population, MPI.COMM_WORLD.Get_size()):
            if i == 0:
                mutations = self.parents[0]
            else:
                mutations = random.choice(self.parents) + ((noise_seed(), stddev),)
            res.append((self.evaluate(mutations, env, trials), mutations))
        sorted_results = sorted([x for batch in MPI.COMM_WORLD.allgather(res) for x in batch],
                                reverse=True)
        self.parents = [x[1] for x in sorted_results][:truncation]
        return sorted_results

    def evaluate(self, mutations, env, trials, step_fn=None):
        """
        Evaluate a genome on an environment.

        Args:
          mutations: a list of (seed, stddev) tuples.
          env: the environment to run.
          trials: the number of episodes to run.
          step_fn: a function to call before each step.

        Returns:
          The mean reward over all the trials.
        """
        with self._noise_adder.seed(mutations):
            self.model.variables_changed()
            rewards = []
            for _ in range(trials):
                done = False
                total_rew = 0.0
                state = self.model.start_state(1)
                obs = env.reset()
                while not done:
                    if step_fn:
                        step_fn()
                    out = self.model.step([obs], state)
                    state = out['states']
                    obs, rew, done, _ = env.step(out['actions'][0])
                    total_rew += rew
                rewards.append(total_rew)
            return sum(rewards) / len(rewards)

def _synchronize_variables(sess, variables):
    if MPI.COMM_WORLD.Get_rank() == 0:
        for var in variables:
            MPI.COMM_WORLD.bcast(sess.run(var))
    else:
        for var in variables:
            sess.run(tf.assign(var, MPI.COMM_WORLD.bcast(None)))
