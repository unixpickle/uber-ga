"""
Various genetic selection algorithms.

A selection algorithm decides parents for a generation
based on fitness values from the previous generation.

All selection algorithms are callables of the form:

    f(population, num_select, **kwargs)

where population is a sequence of (fitness, genome) tuples
and num_select is the number of parents to select.
The return value is a sequence of parent genomes.
Parents may be duplicated in the result, especially since
the pool of parents may be smaller than the requested
number of children.
"""

import random

def truncation_selection(population, children, truncation=10):
    """
    Select the genomes with the highest fitness and sample
    uniformly from this list.

    Args:
      population: the parent population.
      children: the number of children to select.
      truncation: the number of top parents to use.
    """
    sorted_results = sorted(population, reverse=True)
    parents = [x[1] for x in sorted_results][:truncation]
    return [random.choice(parents) for _ in range(children)]

def tournament_selection(population, children, tournament_size=10, choose_prob=1.0):
    """
    Select each child by selecting a random subset of
    parents and using the winner of a "tournament" amongst
    those parents.

    Args:
      population: the parent population.
      children: the number of children to select.
      tournament_size: the number of parents per
        tournament.
      choose_prob: the probability of selecting the top
        performer in a tournament.
    """
    tournament_size = min(tournament_size, len(population))
    res = []
    for _ in range(children):
        players = sorted(random.sample(population, tournament_size), reverse=True)
        res.append(_run_tournament(players, choose_prob)[1])
    return res

def _run_tournament(players, choose_prob):
    for player in players:
        if random.random() <= choose_prob:
            return player
    return players[-1]
