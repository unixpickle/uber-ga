"""
Use a genetic algorithm to learn Pong-v0 in OpenAI Gym.

Run with:

    $ mpiexec -f <your hosts file> python3 -u pong.py

Ideally, you will run this on a fairly large MPI cluster.
"""

# pylint: disable=E1101

import argparse

from anyrl.envs.wrappers import DownsampleEnv, GrayscaleEnv, FrameStackEnv
import gym
from mpi4py import MPI
import tensorflow as tf
from uber_ga import LearningSession, nature_cnn, make_session

def main():
    """
    Train on an Atari game.
    """
    args = parse_args()
    with make_session() as sess:
        env = gym.make(args.env)
        env = FrameStackEnv(DownsampleEnv(GrayscaleEnv(env), 2), 4)
        try:
            model = nature_cnn(sess, env, stochastic=args.stochastic)
            sess.run(tf.global_variables_initializer())
            learn_sess = LearningSession(sess, model)
            while True:
                pop = learn_sess.generation(env,
                                            trials=args.trials,
                                            truncation=args.truncation,
                                            population=args.population,
                                            stddev=args.stddev)
                rewards = [x[0] for x in pop]
                best_rew = learn_sess.evaluate(pop[0][1], env, 1)
                best_rew = MPI.COMM_WORLD.allreduce(best_rew) / MPI.COMM_WORLD.Get_size()
                if MPI.COMM_WORLD.Get_rank() == 0:
                    print('mean=%f best=%f top10=%s' %
                          (sum(rewards)/len(rewards), best_rew, str(rewards[:10])))
                if best_rew >= 20.5:
                    if MPI.COMM_WORLD.Get_rank() == 0:
                        print('Saving video and terminating...')
                        save_video(args, learn_sess, pop[0][1])
                    return
        finally:
            env.close()

def save_video(args, learn_sess, mutations):
    """
    Save a video recording of an agent playing a game.
    """
    env = gym.make(args.env)
    recorder = gym.monitoring.VideoRecorder(env, path='video.mp4')
    env = FrameStackEnv(DownsampleEnv(GrayscaleEnv(env), 2), 4)
    try:
        learn_sess.evaluate(mutations, env, 1, step_fn=recorder.capture_frame)
    finally:
        recorder.close()
        env.close()

def parse_args():
    """
    Parse the command-line arguments.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--stochastic', help='use a stochastic policy', action='store_true')
    parser.add_argument('--trials', help='trials per genome', type=int, default=1)
    parser.add_argument('--truncation', help='top genomes to select', type=int, default=10)
    parser.add_argument('--population', help='genome population', type=int, default=5000)
    parser.add_argument('--stddev', help='mutation stddev', type=float, default=0.1)
    parser.add_argument('env', help='Gym environment ID to run', default='Pong-v0')
    return parser.parse_args()

if __name__ == '__main__':
    main()
