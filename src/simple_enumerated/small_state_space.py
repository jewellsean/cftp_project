from __future__ import division
from numpy.random import RandomState
import pandas as pd
import pyggplot


def markov(rand, state):
    if state == 'B':
        return ('C')
    dart_throw = rand.random_sample()
    if state == 'A' and dart_throw > 0.5:
        return ('B')
    if state == 'C' and dart_throw > 0.5:
        return ('A')
    else:
        return (state)


def mapping(rand, full_state):
    mapping = {}
    for k in full_state:
        mapping[k] = markov(rand, full_state[k])
    return mapping


def constant(full_state):
    if full_state[1] == full_state[2] == full_state[3]:
        return (True)
    else:
        return (False)


def compose(state_t1, state_t2):
    inverse_mapping_values = {'A': 1, 'B': 2, 'C': 3}
    state = {}
    for k in state_t1:
        state[k] = state_t2[inverse_mapping_values[state_t1[k]]]
    return (state)


def couple_from_the_past(rand, full_state):
    while (not constant(full_state)):
        base_state = {1: 'A', 2: 'B', 3: 'C'}
        mapping = mapping(rand, base_state)
        full_state = compose(mapping, full_state)
    return full_state[1]


def couple_forward(rand, full_state):
    while (not constant(full_state)):
        full_state = mapping(rand, full_state)
    return full_state[1]


def couple_engine(rand, cftp, fwd, nSamples):
    full_state = {1: 'A', 2: 'B', 3: 'C'}
    samples = list()
    forward_samples = list()

    for i in range(nSamples):
        full_state_evolution = {1: 'A', 2: 'B', 3: 'C'}
        samples.append(couple_from_the_past(rand, full_state_evolution))
        forward_samples.append(couple_forward(rand, full_state_evolution))

    cftp['A'].append(samples.count('A') / nSamples)
    cftp['B'].append(samples.count('B') / nSamples)
    cftp['C'].append(samples.count('C') / nSamples)

    fwd['A'].append(forward_samples.count('A') / nSamples)
    fwd['B'].append(forward_samples.count('B') / nSamples)
    fwd['C'].append(forward_samples.count('C') / nSamples)


if __name__ == '__main__':

    nSimulations = 10000
    rand = RandomState(1122)

    cftp = {'A': [], 'B': [], 'C': []}
    fwd = {'A': [], 'B': [], 'C': []}

    # couple_from_the_past parameters
    nSamples = 1000

    for simulation in range(nSimulations):
        couple_engine(rand, cftp, fwd, nSamples)

    cftp_data = pd.DataFrame.from_dict(cftp)
    cftp_long = pd.melt(cftp_data, var_name="State", value_name="Proportion")
    p = pyggplot.Plot(cftp_long)
    p.add_boxplot(x='State', y='Proportion')
    p.render("../../figures/cftp_simple_comparison.pdf")

    fwd_data = pd.DataFrame.from_dict(fwd)
    fwd_long = pd.melt(fwd_data, var_name="State", value_name="Proportion")
    p = pyggplot.Plot(fwd_long)
    p.add_boxplot(x='State', y='Proportion')
    p.render("../../figures/fwd_simple_comparison.pdf")