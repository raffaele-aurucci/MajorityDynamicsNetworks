import math
from typing import Callable
import networkx as nx
from tqdm import tqdm

from cost_function import *

# Greedy algorithm
def greedy_seed_set(G: nx.Graph, budget: int, cost_function: Callable) -> set:

    costs = None

    if cost_function.__name__ == 'cost_bridge_capped':
        costs = cost_bridge_capped(G, H=5, tau=20)

        def cost_lookup(G, node):
            return costs[node]

        cost_function = cost_lookup

    degrees = dict(G.degree())
    neighbors = {v: set(G.neighbors(v)) for v in G}

    # Third euristic function
    def euristic(D: set) -> float:
        total = 0.0
        for v, d in degrees.items():
            if d == 0:
                continue
            k = len(neighbors[v].intersection(D))
            for i in range(1, k + 1):
                num = math.ceil(d / 2) - i + 1
                den = d - i + 1
                if num > 0 and den > 0:
                    total += num / den
        return total

    def set_cost(nodes: set) -> int:
        return sum([cost_function(G, node) for node in nodes])

    S_p, S_d = set(), set()
    while set_cost(S_d) <= budget:
        S_p = S_d.copy()
        base_val = euristic(S_d)
        max_node, max_value = None, 0

        for node in tqdm(G.nodes(), desc=f"Iteration with seed_set"):
            if node not in S_d:
                delta = euristic(S_d.union({node})) - base_val
                costo = cost_function(G, node)
                value = delta / costo if costo > 0 else 0
                if value > max_value:
                    max_value, max_node = value, node

        if max_node is not None:
            S_d.add(max_node)
        if S_p == S_d:
            break
        print(S_d)

    return S_d

