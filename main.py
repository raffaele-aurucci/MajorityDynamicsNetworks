import itertools
import io, sys, atexit
import networkx as nx

import algorithms
import cost_function
from influence_diffusion import influence_diffusion

#import matplotlib.pyplot as plt   # se serve la visualizzazione

# Parametri principali
parameters = {
    "fn": [algorithms.WTSS, algorithms.MLPA],
    "K": [32, 64, 128, 320, 640],
    "cost_function": [cost_function.random_cost, cost_function.half_node_degree_cost, cost_function.cost_bridge_capped],
    "euristic": ["f1", "f2", "f3"]  # solo quando fn = algorithms.greedy_seed_set
}

dataset_path = 'datasets/out.as20000102.txt'

# Tee per catturare stdout
class _StdoutTee(io.TextIOBase):
    def __init__(self, original):
        self._original = original
        self._buf = io.StringIO()
    def write(self, s):
        self._buf.write(s)
        return self._original.write(s)
    def flush(self):
        self._original.flush()
    def getvalue(self):
        return self._buf.getvalue()

_original_stdout = sys.stdout
_stdout_tee = _StdoutTee(_original_stdout)
sys.stdout = _stdout_tee

_success = False

def _flush_logs_to_file():
    if not _success:
        return
    try:
        with open('log_route_view.txt', 'a', encoding='utf-8') as f:
            f.write(_stdout_tee.getvalue())
    except Exception:
        pass

atexit.register(_flush_logs_to_file)

# Costruzione grafo
G = nx.Graph()
with open(dataset_path, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            u, v = map(int, parts)
            G.add_edge(u, v)

print("Graph info:\n")

def graph_info(G):
    return (
        f"Type: {'Directed' if G.is_directed() else 'Undirected'}\n"
        f"# Nodes: {G.number_of_nodes()}\n"
        f"# Edges: {G.number_of_edges()}\n"
        f"# Triangles: {sum(nx.triangles(G).values()) // 3}\n"
        f"Highest degree: {max(dict(G.degree()).values())}\n"
        f"Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}\n"
        f"Diameter: {nx.diameter(G)}\n"
        f"Dataset name: {dataset_path.split('/')[1]}\n"
    )

print(graph_info(G))

# Grid search su tutte le combinazioni
for fn, k, cfun in itertools.product(parameters["fn"], parameters["K"], parameters["cost_function"]):
    # caso greedy_seed_set: devo testare con euristiche
    if fn.__name__ == "greedy_seed_set":
        euristics = parameters["euristic"]
    else:
        euristics = [None]

    for heuristic in euristics:
        args = [G, k, cfun]
        if fn.__name__ == "greedy_seed_set" and heuristic is not None:
            print(f"- Function: {fn.__name__} \n- k: {k} \n- Cost function: {cfun.__name__} \n- Euristic function: {heuristic}\n")
            args.append(heuristic)
        else:
            print(f"- Function: {fn.__name__} \n- k: {k} \n- Cost function: {cfun.__name__}\n")

        seed_set = fn(*args)

        print("Seed set lenght:", len(seed_set))
        print("Seed set selected:", seed_set)
        influence_diffusion_set = influence_diffusion(G, seed_set)
        print("Influence diffusion set lenght:", len(influence_diffusion_set))
        print("diffusion ratio:", len(influence_diffusion_set)/G.number_of_nodes())
        print("\n------------------------------------------------\n")

_success = True


