# Majority Dynamics in Networks

The **majority domination problem** is a variant of the classical *dominating set* problem in graph theory. Given a graph \(G=(V,E)\), where \(V\) is the set of vertices and \(E\) the set of edges, a subset \(D \subseteq V\) is a **dominating set** if every vertex not in \(D\) has at least one neighbor in \(D\):

$$
\forall v \in V \setminus D, \quad |N(v) \cap D| \ge 1
$$

where \(N(v)\) denotes the set of neighbors of vertex \(v\).

In **majority domination**, the requirement is stronger: a set \(D \subseteq V\) is a **majority dominating set** if every vertex \(v \in V \setminus D\) has at least half of its neighbors in \(D\). Formally, letting \(d(v) = |N(v)|\) be the degree of \(v\):

$$
\forall v \in V \setminus D, \quad |N(v) \cap D| \ge \left\lceil \frac{d(v)}{2} \right\rceil
$$

Finding a minimum-size majority dominating set is **NP-hard**, making exact solutions impractical for large graphs.

---

Based on this concept, the **Majority Cascade** models influence diffusion in networks. Starting from an initial seed set, nodes activate when a majority of their neighbors are active and remain active permanently. The process ends when a stable state is reached.

When each node has an associated cost (**cost-constrained majority cascade**), the goal is to select a seed set that maximizes activations while respecting a budget \(k\). This problem is **NP-hard** and hard to approximate, often requiring heuristic approaches.

---

In this work, we analyze three approaches for seed set selection under cost-constrained majority cascades:

1. **GREEDY SEED SET** – a standard greedy heuristic from literature.  
2. **WTSS** – a weighted threshold-based approach.  
3. **MLPA** – a novel algorithm proposed in this study.  

We compare their performance across different cost functions and budgets \(k\), evaluating each method’s ability to maximize network diffusion.
