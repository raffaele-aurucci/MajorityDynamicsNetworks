import json
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

import algorithms, cost_function
from influence_diffusion import influence_diffusion

dataset_path = '../dataset/out.as20000102.txt'

# Main parameters for test
parameters = {
    "fn": [algorithms.greedy_seed_set, algorithms.WTSS, algorithms.MLPA],
    "cost_function": [
        cost_function.random_cost,
        cost_function.half_node_degree_cost,
        cost_function.cost_bridge_capped,
    ],
    "euristic": ["f1", "f2", "f3"],  # when fn = algorithms.greedy_seed_set
}

# Percentiles used for computing k values
percentiles = [0.5, 1, 2, 5, 10, 20]


def build_graph(dataset_path):
    """Load graph from dataset file"""
    G = nx.Graph()
    with open(dataset_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                u, v = map(int, parts)
                G.add_edge(u, v)
    return G


def graph_info(G):
    """Return dictionary with basic graph statistics"""
    return {
        "type": "Directed" if G.is_directed() else "Undirected",
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "triangles": sum(nx.triangles(G).values()) // 3,
        "highest_degree": max(dict(G.degree()).values()),
        "average_degree": sum(dict(G.degree()).values()) / G.number_of_nodes(),
        "diameter": nx.diameter(G),
        "average_clustering" : nx.average_clustering(G),
        "dataset_name": dataset_path.split('/')[1],
    }


def compute_k_values(G, cost_fn):
    """Compute k values based on cost function and percentiles"""
    if cost_fn.__name__ == "random_cost":
        n = G.number_of_nodes()
        ks = [max(1, int(n * p / 100)) for p in percentiles]
    elif cost_fn.__name__ == "half_node_degree_cost":
        m = G.number_of_edges()
        ks = [max(1, int(m * p / 100)) for p in percentiles]
    elif cost_fn.__name__ == "cost_bridge_capped":
        costs = cost_fn(G)
        total_cost = sum(costs.values())
        ks = [max(1, int(total_cost * p / 100)) for p in percentiles]
    else:
        ks = [32]  # default fallback
    return ks


def run_experiments(G):
    """Run experiments for all cost functions and algorithms"""
    for cfun in parameters["cost_function"]:
        results = {"experiments": []}
        ks_list = compute_k_values(G, cfun)

        total_experiments = sum(
            len(parameters["euristic"]) if fn.__name__ == "greedy_seed_set" else 1
            for fn in parameters["fn"]
        ) * len(ks_list)

        print(f"\nCost function: {cfun.__name__} | k values: {ks_list}")

        with tqdm(total=total_experiments, desc=f"Running {cfun.__name__}") as pbar:
            filename = f"log_{cfun.__name__}.json"

            for k in ks_list:
                for fn in parameters["fn"]:
                    euristics = parameters["euristic"] if fn.__name__ == "greedy_seed_set" else [None]
                    for heuristic in euristics:
                        args = [G, k, cfun]
                        if fn.__name__ == "greedy_seed_set" and heuristic is not None:
                            args.append(heuristic)

                        seed_set = fn(*args)
                        influence_diffusion_set = influence_diffusion(G, seed_set)

                        exp_result = {
                            "algorithm": fn.__name__,
                            "k": k,
                            "euristic": heuristic,
                            "seed_set_length": len(seed_set),
                            "seed_set": list(seed_set),
                            "influence_diffusion_length": len(influence_diffusion_set),
                            "diffusion_ratio": len(influence_diffusion_set) / G.number_of_nodes()
                        }

                        results["experiments"].append(exp_result)

                        # Save partial results
                        with open(filename, "w", encoding="utf-8") as f:
                            json.dump(results, f, indent=4)

                        pbar.update(1)


def main():

    # Build graph
    G = build_graph(dataset_path)

    # Print graph information
    info = graph_info(G)
    print("\nGraph Info:")
    for k, v in info.items():
        print(f"{k}: {v}")

    # Plot graph with fixed seed layout
    pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(10, 8))
    plt.axis("off")

    # Compute node degrees for coloring
    degrees = dict(G.degree())
    node_colors = [degrees[n] for n in G.nodes()]

    # Draw edges with transparency
    nx.draw_networkx_edges(
        G, pos,
        alpha=0.2,
        edge_color="gray",
        width=0.5
    )

    nodes = nx.draw_networkx_nodes(
        G, pos,
        node_size=40,
        node_color=node_colors,
        cmap=plt.cm.viridis,
        alpha=0.9,
        edgecolors="white",
        linewidths=0.5,
        vmin=0,
        vmax=10
    )

    # Colorbar: tick to 1400 mapped on [0,10]
    cbar = plt.colorbar(nodes)
    cbar.set_label("Node degree", fontsize=10)

    # Ticks each 200
    ticks = range(0, 1460, 200)
    cbar.set_ticks([t / 1460 * 10 for t in ticks])
    cbar.ax.set_yticklabels([str(t) for t in ticks])

    # Title
    plt.title(f"Graph Autonomous systems AS-733", fontsize=14, fontweight="bold")
    plt.show()

    # Run experiments
    # run_experiments(G)


if __name__ == "__main__":
    main()
