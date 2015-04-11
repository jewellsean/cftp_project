import math

import networkx as nx
import numpy as np


def log_sigmoid(x):
    return -1 * math.log(1 + math.exp(- x))


def create_grid(initial_value, grid_size):
    G = nx.Graph()
    nNodes = grid_size ** 2
    nodes = [x for x in range(nNodes)]
    for node in nodes:
        G.add_node(node, value=initial_value)
    for node in nodes:
        # add to right as long as still in correct grid
        if (node % grid_size) != (grid_size - 1):
            G.add_edge(node, node + 1)
        if (node // grid_size < grid_size - 1):
            G.add_edge(node, node + grid_size)
    return G


def graph_to_matrix(G, grid_size):
    attribute_matrix = np.zeros((grid_size, grid_size))
    k = 0
    for i in range(grid_size):
        for j in range(grid_size):
            attribute_matrix[i, j] = G.node[k]['value']
            k = k + 1
    return attribute_matrix


def log_likelihood(nbr_values, temperature):
    return log_sigmoid((2 / temperature) * np.sum(nbr_values))


def check_equality(state1, state2):
    for node in state1.nodes():
        if state1.node[node]['value'] != state2.node[node]['value']:
            return False
    return True