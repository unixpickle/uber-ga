"""
Genetic algorithm outer loop.
"""

# Avoid MPI errors:
# pylint: disable=E1101

import random

from mpi4py import MPI
import tensorflow as tf

from .noise import NoiseSource, NoiseAdder, noise_seeds

class LearningSession:
    """
    A GA optimization session.
    """
    def __init__(self, sess, model, variables=None, noise=None):
        variables = (variables or tf.trainable_variables())
        noise = (noise or NoiseSource())
        self.model = model
        self.parents = [()]
        self._noise_adder = NoiseAdder(sess, variables, noise)
        self._random = random.Random(x=MPI.COMM_WORLD.bcast(random.randint(0, 2**32 - 1)))
        _synchronize_variables(sess, variables)

    def make_offspring(self, population=5000, stddev=0.1):
        """
        Produce a set of offspring from self.parents.
        """
        res = [self.parents[0]]
        for seed in noise_seeds(population - 1):
            parent = self._random.choice(self.parents)
            res.append(parent + ((seed, stddev),))
        return res

    def generation(self, offspring, env, trials=1, truncation=10):
        """
        Run a generation of the algorithm and update
        self.parents for the new generation.

        Call this from each MPI worker.

        Args:
          offspring: the genomes (tuples of seeds) to try
            on the environment.
          env: the gym.Env to use to evaluate the model.
          trials: the number of episodes to run.
          truncation: the number of parents to keep.

        Returns a sorted list of (rew, genome) tuples.
        Updates self.parents to reflect the elite.
        """
        res = {}
        for i in range(MPI.COMM_WORLD.Get_rank(), len(offspring), MPI.COMM_WORLD.Get_size()):
            with self._noise_adder.seed(offspring[i]):
                res[offspring[i]] = self._evaluate(env, trials)
        sorted_results = sorted([(rew, genome)
                                 for batch in MPI.COMM_WORLD.allgather(res)
                                 for genome, rew in batch.items()], reverse=True)
        self.parents = [x[1] for x in sorted_results][:truncation]
        return sorted_results

    def _evaluate(self, env, trials):
        rewards = []
        for _ in range(trials):
            done = False
            total_rew = 0.0
            state = self.model.start_state(1)
            obs = env.reset()
            while not done:
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
