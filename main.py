import networkx as nx
import matplotlib.pyplot as plt
import algorithms
import cost_function
from influence_diffusion import influence_diffusion

G = nx.Graph()

with open('datasets/facebook_combined.txt', 'r') as f:
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

# # Fixed layout with seed
# pos = nx.spring_layout(G, seed=28)
#
# # Visualization graph
# nx.draw(G, pos, node_size=10, with_labels=True, font_size=2)
# plt.show()

seed_set = algorithms.greedy_seed_set(G, 404, cost_function.cost_bridge_capped, 'f1')
print("Seed set selected: ", seed_set)
influence_diffusion_set = influence_diffusion(G, seed_set)
print("Influence diffusion set: ", influence_diffusion_set)
print("len:", len(influence_diffusion_set)/G.number_of_nodes())



