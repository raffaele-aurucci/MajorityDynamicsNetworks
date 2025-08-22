import math
from typing import Callable, Literal, Set, Dict
import networkx as nx
from tqdm import tqdm

from cost_function import *


def heuristic_f1(G: nx.Graph, S: Set[int], degrees: Dict[int, int], neighbors: Dict[int, set]) -> float:
    """
    f1(S) = sum_v min( |N(v) ∩ S| , ceil(d(v)/2) )
    """
    total = 0.0
    for v, d in degrees.items():
        if d == 0:
            continue
        k = len(neighbors[v] & S)
        total += min(k, math.ceil(d / 2))
    return total


def heuristic_f2(G: nx.Graph, S: Set[int], degrees: Dict[int, int], neighbors: Dict[int, set]) -> float:
    """
    f2(S) = sum_v sum_{i=1}^{|N(v)∩S|} max( ceil(d(v)/2) - i + 1 , 0 )
    """
    total = 0.0
    for v, d in degrees.items():
        if d == 0:
            continue
        k = len(neighbors[v] & S)
        t = math.ceil(d / 2)
        # add contributions for i=1..k (capped at 0)
        for i in range(1, k + 1):
            term = t - i + 1
            if term > 0:
                total += term
            else:
                break
    return total


def heuristic_f3(G: nx.Graph, S: Set[int], degrees: Dict[int, int], neighbors: Dict[int, set]) -> float:
    """
    f3(S) = sum_v sum_{i=1}^{|N(v)∩S|} max( (ceil(d(v)/2) - i + 1) / (d(v) - i + 1) , 0 )
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
                # se num<=0 non contribuisce; se den<=0 abbiamo finito (i>d)
                if den <= 0:
                    break
    return total

def greedy_seed_set(
    G: nx.Graph,
    budget: int,
    cost_function: Callable,               # c(u)
    heuristic: Literal["f1", "f2", "f3"] = "f3"
) -> set:
    """
    Implementa l'Algorithm 1 (Cost-Seeds-Greedy) con scelta di f1/f2/f3.
    Ritorna SEMPRE l’ultimo set entro budget.
    """

    # Caso speciale: costi precomputati
    if getattr(cost_function, "__name__", "") == "cost_bridge_capped":
        pre = cost_bridge_capped(G, H=5, tau=20)
        cost_function = lambda G_, u: pre[u]

    # Precompute strutture base
    degrees = dict(G.degree())
    neighbors = {v: set(G.neighbors(v)) for v in G}

    # Selettore euristica
    if heuristic == "f1":
        f = lambda S: heuristic_f1(G, S, degrees, neighbors)
    elif heuristic == "f2":
        f = lambda S: heuristic_f2(G, S, degrees, neighbors)
    elif heuristic == "f3":
        f = lambda S: heuristic_f3(G, S, degrees, neighbors)
    else:
        raise ValueError("heuristic deve essere 'f1', 'f2' o 'f3'.")

    def set_cost(S: Set[int]) -> int:
        return sum(cost_function(G, u) for u in S)

    S_p, S_d = set(), set()
    # loop greedy: aggiungi finché resti entro budget
    while set_cost(S_d) <= budget:
        S_p = S_d.copy()
        rem = budget - set_cost(S_d)

        # candidati ammissibili (non superano il budget residuo)
        candidates = [u for u in G.nodes() if u not in S_d and cost_function(G, u) <= rem]
        if not candidates:
            break

        base = f(S_d)
        best_u, best_val = None, float("-inf")

        for u in candidates:
            delta = f(S_d | {u}) - base
            c = cost_function(G, u)
            # gestione costi nulli (se permessi): priorità massima a delta>0
            score = (delta / c) if c > 0 else (float("inf") if delta > 0 else float("-inf"))
            if score > best_val:
                best_val, best_u = score, u

        if best_u is None:
            break

        S_d.add(best_u)

        # Se per qualunque motivo abbiamo sforato (non dovrebbe succedere con il filtro), restituiamo S_p
        if set_cost(S_d) > budget:
            return S_p

    # qui siamo entro budget
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