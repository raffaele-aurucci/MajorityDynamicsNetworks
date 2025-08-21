import math
import random
import networkx as nx

# Random function
def random_cost(G: nx.Graph , node) -> int:
    return random.randint(1,3)

# Half node degree function
def half_node_degree_cost(G: nx.Graph, node) -> int:
    return math.ceil(G.degree(node) / 2)

# Cost bridge
def cost_bridge_capped(G: nx.Graph, H: int = 5, tau: int = 20) -> dict:
    """
        Computes a cost for each node in a graph based on edge centrality
        and node degree, with a maximum cap and hub penalty.

        The cost reflects the node's importance in the graph connectivity:
        - Nodes with high edge betweenness centrality (on many shortest paths)
          have lower costs,
        - Nodes with low centrality or high-degree hubs receive higher costs,
        - The final cost is bounded between 1 and H.

        Parameters
        ----------
        G : nx.Graph
            The NetworkX graph on which to compute node costs.
        H : int, optional (default=5)
            Maximum cost a node can have.
        tau : int, optional (default=20)
            Degree threshold above which a node is considered a hub
            and incurs a penalty.

        Returns
        -------
        cost : integer costs between 1 and H.

        Notes
        -----
        - Uses normalized edge betweenness centrality.
        - Node betweenness is averaged over incident edges and normalized to [0,1].
        - Final cost formula:
            c = 1 + round((1 - bnorm[u]) * (H - 1)) + hub_penalty
          and is clipped to the range [1, H].
        """
    # 1) normalized edge betweenness
    eb = nx.edge_betweenness_centrality(G, normalized=True)

    # 2) avg betweenness for node
    b = {u: 0.0 for u in G}
    cnt = {u: 0 for u in G}
    for (u, v), val in eb.items():
        b[u] += val
        cnt[u] += 1
        b[v] += val
        cnt[v] += 1
    for u in G:
        b[u] = b[u] / cnt[u] if cnt[u] else 0.0

    # 3) normalized min-max
    vals = list(b.values())
    mn, mx = min(vals), max(vals)
    if mx > mn:
        bnorm = {u: (b[u] - mn) / (mx - mn) for u in G}
    else:
        bnorm = {u: 0.0 for u in G}

    # 4) compute final capped costs
    costs = {}
    for u in G:
        hub_penalty = 1 if G.degree(u) >= tau else 0
        c = 1 + round((1 - bnorm[u]) * (H - 1)) + hub_penalty
        c = max(1, min(H, c))
        costs[u] = int(c)

    return costs