import networkx as nx
import matplotlib.pyplot as plt
import algorithms
import cost_function
from influence_diffusion import influence_diffusion
import sys, io, atexit

# Principal parameters
fn = algorithms.greedy_seed_set
k = 20
cost_function = cost_function.random_cost
heuristic = 'f3'  # imposta a None o rimuovi se non vuoi passare il 4° parametro
dataset_path = 'datasets/ego-facebook.txt'

# Fase preliminare per catturare tutto l'output di print e salvarlo su file alla fine
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

# Installa il "tee" su stdout
_original_stdout = sys.stdout
_stdout_tee = _StdoutTee(_original_stdout)
sys.stdout = _stdout_tee

_success = False  # salva i log solo se True

def _flush_logs_to_file():
    if not _success:
        return
    try:
        with open('log.txt', 'a', encoding='utf-8') as f:
            f.write(_stdout_tee.getvalue())
    except Exception:
        pass

# Registra scrittura su file alla fine dell'esecuzione (solo se _success=True)
atexit.register(_flush_logs_to_file)

# Main code

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
        f"Type: {"Directed" if G.is_directed() else "Undirected"}\n"
        f"# Nodes: {G.number_of_nodes()}\n"
        f"# Edges: {G.number_of_edges()}\n"
        f"# Triangles: {sum(nx.triangles(G).values()) // 3}\n"
        f"Highest degree: {max(dict(G.degree()).values())}\n"
        f"Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}\n"
        f"Diameter: {nx.diameter(G)}\n"
        f"Dataset name: {dataset_path.split("/")[1]}\n"
    )

print(graph_info(G))

# Fixed layout with seed
#pos = nx.spring_layout(G, seed=28)

#print("disegnando grafo...")
# Visualization graph
#nx.draw(G, pos, node_size=10, with_labels=True, font_size=2)
#plt.show()

args = [G, k, cost_function]
if fn.__name__ == "greedy_seed_set" and heuristic is not None:
    print(f"- Function: {fn.__name__} \n- k: {k} \n- Cost function: {cost_function.__name__} \n- Euristic function: {heuristic}\n")
    args.append(heuristic)
else:
    print(f"- Function: {fn.__name__} \n- k: {k} \n- Cost function: {cost_function.__name__}\n")

seed_set = fn(*args)

print("Seed set lenght: ", len(seed_set))
print("Seed set selected: ", seed_set)
influence_diffusion_set = influence_diffusion(G, seed_set)
print("Influence diffusion set lenght: ", len(influence_diffusion_set))
print("diffusion ratio:", len(influence_diffusion_set)/G.number_of_nodes())
print("\n------------------------------------------------\n")

_success = True
