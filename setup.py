"""
Package meta-data.
"""

from setuptools import setup

setup(
    name='uber-ga',
    version='0.0.1',
    description='A distributed genetic algorithm for Reinforcement Learning.',
    long_description='A distributed genetic algorithm for Reinforcement Learning.',
    url='https://github.com/unixpickle/uber-ga',
    author='Alex Nichol',
    author_email='unixpickle@gmail.com',
    license='MIT',
    packages=['uber_ga'],
    install_requires=['anyrl>=0.11.4<0.12.0', 'mpi4py>=2.0.0,<4.0.0']
)
