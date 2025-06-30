from ..optimization import OptimizationSolver
from ..visualization import Visualizer

import networkx as nx

def disprove_demo():
    op = OptimizationSolver()

    # n=12 pair
    T_i = [(5,9),(0,9),(1,4),(0,11),(1,5),(7,8),(2,4),(3,4),(6,8),(5,8),(0,10)]
    T_f = [(6,2),(11,2),(10,7),(11,0),(10,6),(4,3),(9,7),(8,7),(5,3),(6,3),(11,1)]
    n = len(T_i) + 1

    # Node 99 is the new node for the new near-near pair from (0,9) to (2,11)
    nodes = [1, 2, 4, 5, 6, 8, 9, 99]
    edges = [(99, 1, 1), (99, 6, 1), (99, 9, 1), (1, 2, 1), (1, 4, 1), (1, 5, 1), (1, 99, 1), (2, 4, 1), (2, 5, 1), (4, 1, 2), (4, 2, 1), (4, 5, 1), (4, 99, 1), (5, 2, 1), (5, 6, 1), (5, 8, 1), (5, 9, 1), (6, 8, 1), (6, 9, 1), (8, 2, 1), (8, 5, 1), (8, 6, 1), (8, 9, 1), (9, 1, 1), (9, 6, 3), (9, 99, 1)]

    E_i = [(1,4),(2,4),(1,5),(5,8),(6,8),(5,9),(0,10),(0,9)]
    E_f = [(11,1),(6,2),(5,3),(6,3),(10,6),(9,7),(10,7),(2,11)]

    H = nx.DiGraph()
    H.add_nodes_from(nodes)
    for u, v, t in edges:
        H.add_edge(u, v, type=t)

    acyclic_nodes = op.find_largest_acyclic_subgraph(H)
    ac_h = len(acyclic_nodes)
    v_h = len(nodes)
    gamma = None if v_h == 0 else ac_h / v_h

    print(f"Found gamma of: {gamma} = {ac_h}/{v_h} on {n} vertices.")
    print("Nodes included in the subgraph:", acyclic_nodes)

    Visualizer.plot_trees_together(T_i, T_f, "original_trees.png")
    Visualizer.plot_linear_graph(T_i, T_f, E_i, E_f, "linear_graph.png")
    Visualizer.plot_conflict_graph(H, "conflict_graph.png")
    print("Generated all graphs.")


if __name__ == "__main__":
    disprove_demo()
