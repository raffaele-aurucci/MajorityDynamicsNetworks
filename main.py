import math
import random
import time
from typing import Callable
from tqdm import tqdm

import networkx as nx
import matplotlib.pyplot as plt
import scipy

G = nx.Graph()

with open('dataset/out.ego-facebook', 'r') as f:
    next(f)  # salta la prima riga
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            u, v = map(int, parts)
            G.add_edge(u, v)


def graph_info(G):
    return (
        f"Type: {"Directed" if G.is_directed() else "Undirected"}\n"
        f"# Nodes: {G.number_of_nodes()}\n"
        f"# Edges: {G.number_of_edges()}\n"
        f"# Triangles: {sum(nx.triangles(G).values()) // 3}\n"
        f"Highest degree: {max(dict(G.degree()).values())}\n"
        f"Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}\n"
        f"Diameter: {nx.diameter(G)}\n"
    )

print(graph_info(G))

# Fixed layout with seed
pos = nx.spring_layout(G, seed=28)

# Visualization graph
nx.draw(G, pos, node_size=10, with_labels=True, font_size=2)
plt.show()

# Greedy algorithm
def random_cost(graph: nx.Graph , node) -> int:
    return random.randint(1, 4)

def greedy_seed_set(
        graph: nx.Graph,
        budget: int,
        cost_function:Callable
    ) -> set:
    """
    Greedy algorithm for selecting a seed set of nodes in a graph.
    The algorithm iteratively selects nodes that maximize the increase in the
    euristic function while considering the cost of adding each node to the seed set.
    """

    def delta_euristic(euristic, dominating_set, node) -> int:
        return euristic(dominating_set.union({node}), graph) - euristic(dominating_set, graph)
    def set_cost(nodes: set, cost_function: callable) -> int:
        return sum([cost_function(graph, node) for node in nodes])
    def euristic(dominating_set: set, graph: nx.Graph) -> int:
        return sum(
            min(
                len(set(graph.neighbors(node)).intersection(dominating_set)),
                math.ceil(graph.degree(node) / 2)
            )
            for node in graph.nodes()
        )

    # seed set, set of nodes that will be selected
    S_p = set()
    S_d = set()
    while set_cost(S_d, cost_function) <= budget:
        S_p = S_d.copy()
        max_node = None
        max_value = 0
        for node in graph.nodes():
            if node not in S_d:
                value = delta_euristic(euristic, S_d, node)
                value /= cost_function(graph, node)
                if value > max_value:
                    max_value = value
                    max_node = node
        S_d.add(max_node)
    return S_p


# start = time.time()
# seed_set = greedy_seed_set(graph=G, budget=100, cost_function=random_cost)
# end = time.time()

# print(f"Execution time: {end - start:.4f} seconds")
# print(seed_set)
# print(len(seed_set))