"""
Use a genetic algorithm to learn CartPole-v0.

Run with:

    $ mpirun -n 8 python3 -u cartpole.py

You can change `8` to any value. It effects speed, but not
sample efficiency.
"""

import gym
import tensorflow as tf
from uber_ga import LearningSession, simple_mlp

POPULATION = 100

def main():
    """
    Train on CartPole.
    """
    with tf.Session() as sess:
        env = gym.make('CartPole-v0')
        try:
            model = simple_mlp(sess, env)
            sess.run(tf.global_variables_initializer())
            learn_sess = LearningSession(sess, model)
            while True:
                offspring = learn_sess.make_offspring(population=POPULATION)
                pop = learn_sess.generation(offspring, env, trials=5)
                rewards = [x[0] for x in pop]
                print('mean=%f best=%s' % (sum(rewards)/len(rewards), str(rewards[:10])))
        finally:
            env.close()

if __name__ == '__main__':
    main()
