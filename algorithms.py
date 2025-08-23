import math
from typing import Callable, Literal, Set, Dict
import networkx as nx
from tqdm import tqdm
import random
from cost_function import *


def _heuristic_f1(G: nx.Graph, S: Set[int], degrees: Dict[int, int], neighbors: Dict[int, set]) -> float:
    """
    Compute heuristic f1(S) = sum_v min(|N(v) ∩ S|, ceil(d(v)/2)).

    Args:
        G: NetworkX graph
        S: set of selected seed nodes
        degrees: dictionary of node degrees
        neighbors: dictionary of node neighbor sets

    Returns:
        float heuristic value
    """
    total = 0.0
    for v, d in degrees.items():
        if d == 0:
            continue
        k = len(neighbors[v] & S)
        total += min(k, math.ceil(d / 2))
    return total


def _heuristic_f2(G: nx.Graph, S: Set[int], degrees: Dict[int, int], neighbors: Dict[int, set]) -> float:
    """
    Compute heuristic f2(S) = sum_v sum_{i=1}^{|N(v)∩S|} max(ceil(d(v)/2) - i + 1, 0).
    """
    total = 0.0
    for v, d in degrees.items():
        if d == 0:
            continue
        k = len(neighbors[v] & S)
        t = math.ceil(d / 2)
        for i in range(1, k + 1):
            term = t - i + 1
            if term > 0:
                total += term
            else:
                break
    return total


def _heuristic_f3(G: nx.Graph, S: Set[int], degrees: Dict[int, int], neighbors: Dict[int, set]) -> float:
    """
    Compute heuristic f3(S) = sum_v sum_{i=1}^{|N(v)∩S|} max((ceil(d(v)/2) - i + 1)/(d(v) - i + 1), 0).
    """
    total = 0.0
    for v, d in degrees.items():
        if d == 0:
            continue
        k = len(neighbors[v] & S)
        for i in range(1, k + 1):
            num = math.ceil(d / 2) - i + 1
            den = d - i + 1
            if num > 0 and den > 0:
                total += num / den
            else:
                if den <= 0:
                    break
    return total


def greedy_seed_set(G: nx.Graph, budget: int, cost_function: Callable, heuristic: Literal["f1", "f2", "f3"] = "f3") -> set:
    """
    Greedy algorithm to select seed set within budget using chosen heuristic.

    Args:
        G: NetworkX graph
        budget: maximum total cost allowed
        cost_function: callable function to compute node cost
        heuristic: one of 'f1', 'f2', 'f3'

    Returns:
        Set of selected seed nodes within budget
    """

    # Handle precomputed cost function
    if cost_function.__name__ == 'cost_bridge_capped':
        precomputed_costs = cost_bridge_capped(G, H=5, tau=20)
        cost_function = lambda G, node: precomputed_costs[node]

    degrees = dict(G.degree())
    neighbors = {v: set(G.neighbors(v)) for v in G}

    # Select heuristic function
    if heuristic == "f1":
        f = lambda S: _heuristic_f1(G, S, degrees, neighbors)
    elif heuristic == "f2":
        f = lambda S: _heuristic_f2(G, S, degrees, neighbors)
    elif heuristic == "f3":
        f = lambda S: _heuristic_f3(G, S, degrees, neighbors)
    else:
        raise ValueError("heuristic must be 'f1', 'f2' or 'f3'.")

    def set_cost(S: Set[int]) -> int:
        return sum(cost_function(G, u) for u in S)

    S_p, S_d = set(), set()

    # Greedy loop: add nodes until budget is exceeded
    while set_cost(S_d) <= budget:
        S_p = S_d.copy()
        rem = budget - set_cost(S_d)
        candidates = [u for u in G.nodes() if u not in S_d and cost_function(G, u) <= rem]
        if not candidates:
            break

        base = f(S_d)
        best_u, best_val = None, float("-inf")

        for u in candidates:
            delta = f(S_d | {u}) - base
            c = cost_function(G, u)
            score = (delta / c) if c > 0 else (float("inf") if delta > 0 else float("-inf"))
            if score > best_val:
                best_val, best_u = score, u

        if best_u is None:
            break

        S_d.add(best_u)

        if set_cost(S_d) > budget:
            return S_p

    return S_d


def _LPA_partition(G, max_iter=20):
    """
    Asynchronous Label Propagation Algorithm to detect communities.

    Args:
        G: NetworkX graph
        max_iter: maximum number of iterations

    Returns:
        List of sets of nodes representing communities
    """
    labels = {v: v for v in G.nodes()}
    nodes = list(G.nodes())
    for _ in range(max_iter):
        random.shuffle(nodes)
        changed = False
        for v in nodes:
            if not list(G.neighbors(v)):
                continue
            counts = {}
            for u in G.neighbors(v):
                lbl = labels[u]
                counts[lbl] = counts.get(lbl, 0) + 1
            max_count = max(counts.values())
            best = [lbl for lbl, c in counts.items() if c == max_count]
            new_label = random.choice(best)
            if labels[v] != new_label:
                labels[v] = new_label
                changed = True
        if not changed:
            break
    comms = {}
    for v, lbl in labels.items():
        comms.setdefault(lbl, set()).add(v)
    return list(comms.values())


def _closed_nei(G, v):
    """Return closed neighborhood of a node (node itself + neighbors)."""
    return set(G.neighbors(v)) | {v}


def MLPA(G, budget: int, cost_function: Callable):
    """
    MLPA algorithm to select seed set within budget using community structure.

    Args:
        G: NetworkX graph
        budget: maximum allowed total cost
        cost_function: callable node cost function

    Returns:
        Set of selected seed nodes
    """

    if cost_function.__name__ == 'cost_bridge_capped':
        precomputed_costs = cost_bridge_capped(G, H=5, tau=20)
        cost_function = lambda G, node: precomputed_costs[node]

    cost = {v: cost_function(G, v) for v in G.nodes()}
    communities = _LPA_partition(G)
    t = {v: math.ceil(G.degree(v) / 2) for v in G.nodes()}  # majority thresholds
    delta = {v: t[v] for v in G.nodes()}  # deficits
    S = set()
    total_cost = 0

    def gain(u):
        return sum(1 for w in _closed_nei(G, u) if delta[w] > 0)

    while any(delta[v] > 0 for v in G.nodes()) and total_cost < budget:
        random.shuffle(communities)
        progress = False
        for C in communities:
            candidates = [v for v in C if v not in S and total_cost + cost[v] <= budget]
            if not candidates:
                continue
            u = max(candidates, key=lambda v: (gain(v) / max(1, cost[v]), G.degree(v)))
            if gain(u) == 0:
                continue
            if total_cost + cost[u] <= budget:
                S.add(u)
                total_cost += cost[u]
                for w in _closed_nei(G, u):
                    if delta[w] > 0:
                        delta[w] -= 1
                progress = True
        if not progress:
            break

    return S
