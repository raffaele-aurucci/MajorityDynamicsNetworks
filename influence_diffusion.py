import networkx as nx


def influence_diffusion(G: nx.Graph, seed_set: set) -> set:
    """
    Simulate majority-based influence diffusion in a graph.

    Starting from the initial seed set, nodes become influenced if more than
    half of their neighbors are already influenced. The process iterates
    until no new nodes are influenced.

    Args:
        G: NetworkX graph
        seed_set: initial set of seed nodes

    Returns:
        Set of all influenced nodes after diffusion stabilizes
    """
    influenced_nodes = set(seed_set)

    while True:
        new_influenced_nodes = set(influenced_nodes)
        for node in influenced_nodes:
            for neighbor in G.neighbors(node):
                if neighbor not in influenced_nodes:
                    # Check if more than half of neighbor nodes are influenced
                    neighbors = set(G.neighbors(neighbor))
                    influenced_neighbors = len(neighbors & influenced_nodes)
                    if influenced_neighbors > (G.degree(neighbor) / 2):
                        new_influenced_nodes.add(neighbor)
        # Stop if no new nodes are influenced
        if new_influenced_nodes == influenced_nodes:
            break
        influenced_nodes = new_influenced_nodes

    return influenced_nodes

