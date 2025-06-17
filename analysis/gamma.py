import networkx as nx
from typing import List, Tuple, Optional, Dict, Any

from .conflict import ConflictAnalyzer
from ..optimization import OptimizationSolver
from ..visualization import Visualizer
from ..core import TreeUtils
from ..generation import NCSTGenerator


class GammaAnalyzer:
    """Main analyzer for computing gamma values between tree pairs."""

    def __init__(self):
        self.conflict_analyzer = ConflictAnalyzer()
        self.optimizer = OptimizationSolver()
        self.visualizer = Visualizer()

    def analyze_tree_pair(
        self, T_i: List[Tuple[int, int]],
        T_f: List[Tuple[int, int]],
        verbose: bool = True,
        plot: bool = True
    ) -> Tuple[Optional[float], int, List[Tuple[int, int]], List[Tuple[int, int]], nx.DiGraph]:
        """Analyze a pair of trees and compute gamma value."""
        n = len(T_i) + 1

        # Get conflict vertices and edge pairs
        conflict_vertices, E_i, E_f = self.conflict_analyzer.get_gaps_and_edge_pairs(
            T_i, T_f)

        # Get conflict edges
        conflict_edges = self.conflict_analyzer.get_conflict_edges(
            conflict_vertices, E_i, E_f)

        # Create conflict graph
        H = nx.DiGraph()
        H.add_nodes_from(conflict_vertices)
        for u, v, t in conflict_edges:
            H.add_edge(u, v, type=t)

        # Find largest acyclic subgraph
        acyclic_nodes = self.optimizer.find_largest_acyclic_subgraph(H)
        ac_h = len(acyclic_nodes)
        v_h = len(conflict_vertices)
        gamma = None if v_h == 0 else ac_h / v_h

        if verbose:
            print(f"Found gamma of: {gamma} = {ac_h}/{v_h} on {n} vertices.")
            print("Nodes included in the subgraph:", acyclic_nodes)

        if gamma is not None and gamma < 4/9:
            print("✅✅✅ FOUND BETTER PAIR OF TREES ✅✅✅")
            print("T_i:", T_i)
            print("T_f:", T_f)

        if plot:
            self.visualizer.plot_trees_together(T_i, T_f, "original_trees.png")
            self.visualizer.plot_linear_graph(
                T_i, T_f, E_i, E_f, "linear_graph.png")
            self.visualizer.plot_conflict_graph(H, "conflict_graph.png")

        return gamma, ac_h, E_i, E_f, H

    def search_for_gamma_threshold(self, n: int,
                                   gamma_threshold: float,
                                   method: str = "random",
                                   k: Optional[int] = None,
                                   verbose: bool = True,
                                   plot: bool = True) -> Dict[str, Any]:
        """Search for tree pairs meeting gamma threshold."""
        curr_gamma = ac_h = v_h = 99
        seed_i = seed_f = 0
        T_i = T_f = E_i = E_f = []
        H = nx.DiGraph()
        num_tested = 0

        while curr_gamma is None or curr_gamma > gamma_threshold:
            # Generate first tree
            if k is None:
                T_i, seed_i = NCSTGenerator.generate_random_ncst(n)
            else:
                T_i, seed_i = NCSTGenerator.generate_ncst_with_k_borders(n, k)

            num_tested += 1

            # Generate second tree based on method
            if method == "rf":
                T_f = TreeUtils.flip_tree(TreeUtils.rotate_tree(T_i, n // 2))
                seed_f = method
            elif method == "fr":
                T_f = TreeUtils.rotate_tree(TreeUtils.flip_tree(T_i), n // 2)
                seed_f = method
            elif method == "f":
                T_f = TreeUtils.flip_tree(T_i)
                seed_f = method
            elif method == "r":
                T_f = TreeUtils.rotate_tree(T_i, n // 2)
                seed_f = method
            else:  # random
                if k is None:
                    T_f, seed_f = NCSTGenerator.generate_random_ncst(n)
                else:
                    T_f, seed_f = NCSTGenerator.generate_ncst_with_k_borders(
                        n, k)

            curr_gamma, ac_h, E_i, E_f, H = self.analyze_tree_pair(
                T_i, T_f, verbose=False, plot=False)
            v_h = len(H.nodes)

            # Print notable results
            if curr_gamma is not None and (curr_gamma < gamma_threshold or verbose or
                                           (curr_gamma < 0.6)):
                print(
                    f"Found gamma of: {curr_gamma} = {
                        ac_h}/{v_h} on {n} vertices "
                    f"with seeds {seed_i} and {seed_f}."
                )

        if plot and curr_gamma is not None and curr_gamma <= gamma_threshold:
            self.visualizer.plot_trees_together(T_i, T_f, "original_trees.png")
            self.visualizer.plot_linear_graph(
                T_i, T_f, E_i, E_f, "linear_graph.png")
            self.visualizer.plot_conflict_graph(H, "conflict_graph.png")

        return {
            'gamma': curr_gamma,
            'T_i': T_i,
            'T_f': T_f,
            'seed_i': seed_i,
            'seed_f': seed_f,
            'num_tested': num_tested,
            'H': H,
            'E_i': E_i,
            'E_f': E_f
        }

    def analyze_pair_blowup(self, T_i: List[Tuple[int, int]], T_f: List[Tuple[int, int]], k: int, verbose: bool = True, plot: bool = True):
        """Function to test the k blowup of two trees"""

        # Get the vertices of H = H(T,T') AND the near-near edge pairs associated
        conflict_vertices, E_i, E_f = self.conflict_analyzer.get_gaps_and_edge_pairs(T_i, T_f)

        # Get the k blowups
        T_i_blown = TreeUtils.blowup_tree(T_i, k, E_i, conflict_vertices)
        T_f_blown = TreeUtils.blowup_tree(T_f, k, E_f, conflict_vertices)

        self.analyze_tree_pair(T_i_blown, T_f_blown, verbose, plot)
