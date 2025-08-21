import math
from typing import Callable
import networkx as nx
from tqdm import tqdm

from cost_function import *

# Greedy algorithm
def greedy_seed_set(G: nx.Graph, budget: int, cost_function: Callable) -> set:

    # Handle special case for cost function
    if cost_function.__name__ == 'cost_bridge_capped':
        precomputed_costs = cost_bridge_capped(G, H=5, tau=20)
        cost_function = lambda G, node: precomputed_costs[node]

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



def WTSS(G: nx.Graph, budget: int, cost_function: Callable) -> set:
    """
    Budget-constrained variant of WTSS.

    Input:
        G: NetworkX graph
        budget: maximum allowed total cost
        cost_function: function computing the cost of a node
    Output:
        S: selected target set
    """

    # --- Initialization ---
    S = set()
    total_cost = 0
    U = set(G.nodes())

    # Handle special case for cost function
    if cost_function.__name__ == 'cost_bridge_capped':
        precomputed_costs = cost_bridge_capped(G, H=5, tau=20)
        cost_function = lambda G, node: precomputed_costs[node]

    # Data structures
    cost = {v: cost_function(G, v) for v in U}
    delta = {v: G.degree(v) for v in U}
    k = {v: math.ceil(G.degree(v) / 2) for v in U}
    N = {v: set(G.neighbors(v)) for v in U}

    def case1_update(v):
        """ Case 1: Node v is activated by neighbors. """
        for u in N[v]:
            if u in U:
                k[u] = max(k[u] - 1, 0)

    def case2_update(v):
        """ Case 2: Node v is added to S. """
        nonlocal total_cost
        S.add(v)
        total_cost += cost[v]
        for u in N[v]:
            if u in U:
                k[u] -= 1

    def case3_select():
        """ Case 3: Select node maximizing (c(v) * k(v)) / (δ(v) * (δ(v)+1)) """
        best_node, max_value = None, -1
        for v in U:
            if delta[v] > 0 and total_cost + cost[v] <= budget:
                value = (cost[v] * k[v]) / (delta[v] * (delta[v] + 1))
                if value > max_value:
                    best_node, max_value = v, value
        return best_node


    # Main loop
    while U and total_cost <= budget:
        node = None

        # Case 1
        node = next((v for v in U if k[v] == 0), None)
        if node:
            case1_update(node)

        # Case 2
        elif any(delta[v] < k[v] for v in U):
            for v in U:
                if delta[v] < k[v] and total_cost + cost[v] <= budget:
                    node = v
                    case2_update(node)
                    break

        # Case 3
        else:
            node = case3_select()

        if not node:  # No valid node → stop
            break

        # Update delta and neighbor sets for all cases
        for u in N[node]:
            if u in U:
                delta[u] -= 1
                N[u].discard(node)

        # Remove node from U
        U.remove(node)

    return S