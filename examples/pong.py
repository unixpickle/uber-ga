"""
Use a genetic algorithm to learn Pong-v0 in OpenAI Gym.

Run with:

    $ mpiexec -f <your hosts file> python3 -u pong.py

Ideally, you will run this on a fairly large MPI cluster.
"""

from anyrl.envs.wrappers import DownsampleEnv, GrayscaleEnv, FrameStackEnv
import gym
from mpi4py import MPI
import tensorflow as tf
from uber_ga import LearningSession, nature_cnn

def main():
    """
    Train on CartPole.
    """
    with tf.Session() as sess:
        env = gym.make('Pong-v0')
        env = FrameStackEnv(DownsampleEnv(GrayscaleEnv(env), 2), 4)
        try:
            model = nature_cnn(sess, env)
            sess.run(tf.global_variables_initializer())
            learn_sess = LearningSession(sess, model)
            while True:
                pop = learn_sess.generation(env)
                rewards = [x[0] for x in pop]
                if MPI.COMM_WORLD.Get_rank() == 0: # pylint: disable=E1101
                    print('mean=%f best=%s' % (sum(rewards)/len(rewards), str(rewards[:10])))
        finally:
            env.close()

if __name__ == '__main__':
    main()
