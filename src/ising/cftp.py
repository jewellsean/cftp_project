import pandas as pd
from numpy.random import RandomState

import ising_utils as ut
import ising_kernels as kn


def monotone_couple_past(rand, grid_size, temperature):
    T = 1
    upper = ut.create_grid(1, grid_size)
    lower = ut.create_grid(-1, grid_size)
    rand_start = rand.get_state()

    while (not ut.check_equality(upper, lower)):
        rand.set_state(rand_start)
        seeds = rand.uniform(0, 4294967295, T)
        for i in range(T):
            rand.seed(int(seeds[i]))
            kn.full_gibbs_step(rand, upper, temperature)
            rand.seed(int(seeds[i]))
            kn.full_gibbs_step(rand, lower, temperature)
        T *= 2
    return lower


def couple_engine(rand, nSamples, grid_size, temperature, results):
    seeds = rand.uniform(0, 4294967295, nSamples)
    samples = {}
    samples['init_value'] = list()
    samples['number_components'] = list()
    for i in range(nSamples):
        rand.seed(int(seeds[i]))
        sample = monotone_couple_past(rand, grid_size, temperature)
        kn.process_sample(sample, samples)

    results['prob_estimate'].append(np.mean(samples['init_value']))
    results['component_estimate'].append(np.mean(samples['number_components']))
    results['temperature'].append(temperature)
    results['grid_size'].append(grid_size)


if __name__ == '__main__':
    rand = RandomState(45123)
    nSamples = 1000
    temperatures = [10, 100, 1000]
    grid_sizes = [10, 50, 100]

    results = {'prob_estimate': [], 'component_estimate': [], 'temperature': [], 'grid_size': []}

    for grid_size in grid_sizes:
        for temperature in temperatures:
            couple_engine(rand, nSamples, grid_size, temperature, results)
    data = pd.DataFrame.from_dict(results)
    data.to_csv("../../results/monotone_cftp.csv")


