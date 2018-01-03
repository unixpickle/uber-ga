"""
Genetic algorithm outer loop.
"""

# Avoid MPI errors:
# pylint: disable=E1101

from mpi4py import MPI
import tensorflow as tf

from .noise import NoiseSource, NoiseAdder, noise_seed
from .selection import tournament_selection

class LearningSession:
    """
    A GA optimization session.
    """
    def __init__(self, session, model, noise=None, selection=tournament_selection):
        self.session = session
        self.model = model
        self.population = None
        self.selection = selection
        self._noise_adder = NoiseAdder(self.session, self.model.variables, noise or NoiseSource())
        _synchronize_variables(self.session, self.model.variables)

    def export_state(self):
        """
        Export the state of the learning session to a
        picklable object.
        """
        return {
            'model': self.model.export_state(),
            'population': self.population,
            'noise': self._noise_adder.noise
        }

    def import_state(self, state):
        """
        Import a state exported by export_state().

        This assumes that the LearningSession has already
        been setup with a suitable TensorFlow graph.

        This may add nodes (e.g. assigns) to the graph, so
        it should not be called often.
        """
        self.model.import_state(state['model'])
        self.population = state['population']
        self._noise_adder.noise = state['noise']

    # pylint: disable=R0913
    def generation(self, env, trials=1, population=5000, stddev=0.1, **select_kwargs):
        """
        Run a generation of the algorithm and update the
        population accordingly.

        Call this from each MPI worker.

        Args:
          env: the gym.Env to use to evaluate the model.
          trials: the number of episodes to run.
          population: the number of new genomes to try.
          stddev: mutation standard deviation.
          select_kwargs: kwargs for selection algorithm.

        Updates self.population to a sorted list of
        (fitness, genome) tuples.
        """
        selected = self._select(population, select_kwargs)
        res = []
        for i in range(MPI.COMM_WORLD.Get_rank(), population+1, MPI.COMM_WORLD.Get_size()):
            if i == 0 and self.population is not None:
                mutations = self.population[0]
            else:
                mutations = selected[i - 1] + ((noise_seed(), stddev),)
            res.append((self.evaluate(mutations, env, trials), mutations))
        full_res = [x for batch in MPI.COMM_WORLD.allgather(res) for x in batch]
        self.population = sorted(full_res, reverse=True)

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

    def _select(self, children, select_kwargs):
        if MPI.COMM_WORLD.Get_rank() == 0:
            if self.population is None:
                selected = [()] * children
            else:
                selected = self.selection(self.population, children, **select_kwargs)
            MPI.COMM_WORLD.bcast(selected)
            return selected
        return MPI.COMM_WORLD.bcast(None)

def _synchronize_variables(sess, variables):
    if MPI.COMM_WORLD.Get_rank() == 0:
        for var in variables:
            MPI.COMM_WORLD.bcast(sess.run(var))
    else:
        for var in variables:
            sess.run(tf.assign(var, MPI.COMM_WORLD.bcast(None)))
