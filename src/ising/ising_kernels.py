import math

import networkx as nx

import ising_utils as ising


def full_gibbs_step(rand, state, temperature):
    nodes = state.nodes()
    for node in nodes:
        node_gibbs_step(rand, node, state, temperature)


def node_gibbs_step(rand, node, state, temperature):
    node_nbrs = state.neighbors(node)
    nbr_values = [state.node[idx]['value'] for idx in node_nbrs]
    up_log_likelihood = ising.log_likelihood(nbr_values, temperature)
    if rand.uniform() <= math.exp(up_log_likelihood):
        state.node[node]['value'] = 1
    else:
        state.node[node]['value'] = -1


def number_of_connected_components(state):
    K = nx.Graph(state)
    for e in K.edges():
        if ( K.node[e[0]]['value'] != K.node[e[1]]['value']):
            K.remove_edge(*e)
    return (nx.number_connected_components(K))


def process_sample(state, samples):
    samples['init_value'].append((state.node[0]['value'] + 1) / 2)
    samples['number_components'].append(number_of_connected_components(state))