import math
from typing import Callable

import networkx as nx
from cost_function import *

# Greedy algorithm
def greedy_seed_set(G: nx.Graph, budget: int, cost_function:Callable) -> set:
    """
        Selects a seed set of nodes in a graph using a greedy algorithm.

        The algorithm iteratively adds the node that maximizes the increase in a
        heuristic function (e.g., influence, coverage, diffusion) per unit cost,
        until the available budget is exhausted.

        Parameters
        ----------
        G : nx.Graph
            The graph on which to select seed nodes. Can be directed or undirected.
        budget : int
            The maximum number of nodes or total allowed cost for the seed set.
        cost_function : Callable
            A function that takes a node as input and returns its associated cost.
            Used to weigh the addition of each node to the seed set based on its cost.

        Returns
        -------
        set
            A set of nodes selected as the seed set.
    """

    def delta_euristic(euristic: callable, D: set, node) -> int:
        return euristic(D.union({node}), G) - euristic(D, G)

    def set_cost(nodes: set, cost_function: callable) -> int:
        return sum([cost_function(G, node) for node in nodes])

    def euristic(D: set, G: nx.Graph) -> float:
        """
            Computes f3(D) for an undirected graph G and a dominating set D.

            f3(D) = Σ_{v ∈ V} Σ_{i=1}^{|N(v)∩D|} max{ (ceil(d(v)/2) - i + 1) / (d(v) - i + 1), 0 }

            Parameters
            ----------
            G : nx.Graph
                An undirected graph (networkx)
            D : list or set
                The set of selected nodes

            Returns
            -------
            float
                The value of the function f3(D)
        """
        total = 0.0

        for v in G.nodes():
            d = G.degree(v)
            if d == 0:
                continue  # isolated node

            # neighbours of v in D
            k = sum((u in D) for u in G.neighbors(v))

            # sum of decreasing and normalized contributions
            for i in range(1, k + 1):
                num = math.ceil(d / 2) - i + 1
                den = d - i + 1
                contrib = num / den if num > 0 and den > 0 else 0.0
                total += contrib

        return total


    # seed set, set of nodes that will be selected
    S_p = set()
    S_d = set()
    while set_cost(S_d, cost_function) <= budget:
        S_p = S_d.copy()
        max_node = None
        max_value = 0
        for node in G.nodes():
            if node not in S_d:
                value = delta_euristic(euristic, S_d, node)
                value /= cost_function(G, node)
                if value > max_value:
                    max_value = value
                    max_node = node
        S_d.add(max_node)
        print(S_p)
    return S_p