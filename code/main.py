import json
import networkx as nx
import algorithms, cost_function
from influence_diffusion import influence_diffusion
from tqdm import tqdm

dataset_path = '../datasets/out.as20000102.txt'

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

# Percentiles
percentiles = [0.5, 1, 2, 5, 10, 20]

# Build of graphs
G = nx.Graph()
with open(dataset_path, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            u, v = map(int, parts)
            G.add_edge(u, v)

def graph_info(G):
    return {
        "type": "Directed" if G.is_directed() else "Undirected",
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "triangles": sum(nx.triangles(G).values()) // 3,
        "highest_degree": max(dict(G.degree()).values()),
        "average_degree": sum(dict(G.degree()).values()) / G.number_of_nodes(),
        "diameter": nx.diameter(G),
        "dataset_name": dataset_path.split('/')[1],
    }

def compute_k_values(G, cost_fn):
    if cost_fn.__name__ == "random_cost":
        n = G.number_of_nodes()
        ks = [max(1, int(n * p / 100)) for p in percentiles]
    elif cost_fn.__name__ == "half_node_degree_cost":
        m = G.number_of_edges()
        ks = [max(1, int(m * p / 100)) for p in percentiles]
    elif cost_fn.__name__ == "cost_bridge_capped":
        # Cost's of node (H = max 5)
        costs = cost_fn(G)
        total_cost = sum(costs.values())
        ks = [max(1, int(total_cost * p / 100)) for p in percentiles]
    else:
        ks = [32]  # default fallback
    return ks

# --- Loop for each cost function ---
for cfun in parameters["cost_function"]:
    results = {"experiments": []}

    ks_list = compute_k_values(G, cfun)
    total_experiments = sum(len(parameters["euristic"]) if fn.__name__=="greedy_seed_set" else 1 for fn in parameters["fn"]) * len(ks_list)

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

                    # save
                    with open(filename, "w", encoding="utf-8") as f:
                        json.dump(results, f, indent=4)
