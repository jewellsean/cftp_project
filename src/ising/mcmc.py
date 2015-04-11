from __future__ import division

import networkx as nx
from numpy.random import RandomState
import numpy as np
import pandas as pd

import ising_kernels as kn


def mcmc_engine(rand, grid_size, temperature, nMCMC, thin, burn, results):
    G = nx.Graph()

    nNodes = grid_size ** 2
    nodes = [x for x in range(nNodes)]
    for node in nodes:
        G.add_node(node, value=rand.choice([1, -1]))

    for node in nodes:
        # add to right as long as still in correct grid
        if (node % grid_size) != (grid_size - 1):
            G.add_edge(node, node + 1)
        if (node // grid_size < grid_size - 1):
            G.add_edge(node, node + grid_size)

    samples = {}
    samples['init_value'] = list()
    samples['number_components'] = list()
    for i in range(nMCMC):
        kn.full_gibbs_step(rand, G, temperature)
        if (i % thin == 0 and i > burn):
            kn.process_sample(G, samples)

    results['prob_estimate'].append(np.mean(samples['init_value']))
    results['component_estimate'].append(np.mean(samples['number_components']))
    results['temperature'].append(temperature)
    results['grid_size'].append(grid_size)


if __name__ == '__main__':
    rand = RandomState(45456431)

    nMCMC = 10000
    thin = 10
    burn = 1000

    temperatures = [10, 100, 1000]
    grid_sizes = [10, 50, 100]

    results = {'prob_estimate': [], 'component_estimate': [], 'temperature': [], 'grid_size': []}

    for grid_size in grid_sizes:
        for temperature in temperatures:
            mcmc_engine(rand, grid_size, temperature, nMCMC, thin, burn, results)

    data = pd.DataFrame.from_dict(results)
    data.to_csv("../../results/mcmc_grid.csv")

