import networkx as nx
from typing import List, Tuple

from ..analysis import GammaAnalyzer
from ..visualization import Visualizer
from ..formulas import NCSTFormulas
from ..generation import NecklaceGenerator
from ..core import TreeUtils, UnionFind


class FiniteGammaSearcher:
    """Enumerative searcher for all NCSTs, with restrictions"""

    def __init__(self):
        self.gamma_analyzer = GammaAnalyzer()
        self.visualizer = Visualizer()

    def enumerate_ncsts_k_borders(self, n: int, k: int, test: bool = True):
        """Function to enumerate over all NCSTs with exactly k borders

            PARAMETERS:
                n: number of vertices
                k: number of borders
        """
        seen = set()
        num_tested = 0
        best_gamma = 1
        best_trees = [], []
        four_nine_gamma_counter = 0
        four_nine_list = []

        print(f"Beginning test on all NCSTSs with {n} vertices and {k} borders")
        num_trees = NCSTFormulas.T(n, k)
        Visualizer.print_progress_bar(num_tested, num_trees)

        def is_valid_tree(points, edges):
            # A graph is a tree iff it is connected and has n-1 edges
            if len(edges) != len(points) - 1:
                return False
            G = nx.Graph()
            G.add_nodes_from(points)
            G.add_edges_from(edges)
            return nx.is_connected(G)

        def has_cycle_uf(edges):
            uf = UnionFind()
            for u, v in edges:
                if uf[u] == uf[v]:
                    return True
                uf.union(u, v)
            return False

        def enumerate_ncsts_helper(self, points, local_edges, all_edges):
            nonlocal num_tested, seen, best_gamma, best_trees, four_nine_gamma_counter


            if len(all_edges) == n - 1 and is_valid_tree(range(n), all_edges):
                base_tree = [(min(a, b), max(a, b)) for a, b in all_edges]
                flipped_tree = TreeUtils.flip_tree(base_tree)

                def rotated_versions(tree):
                    return [
                        frozenset(
                            (min((a + r) % n, (b + r) % n), max((a + r) % n, (b + r) % n))
                            for a, b in tree
                        )
                        for r in range(n)
                    ]

                for tree_variant in (base_tree, flipped_tree):
                    for rotated in rotated_versions(tree_variant):
                        if rotated in seen:
                            continue
                        seen.add(rotated)
                        num_tested += 1

                        to_test: List[Tuple[int, int]] = list(rotated)

                        gamma = None
                        if test:
                            gamma, ac_h, E_i, E_f, H = self.gamma_analyzer.analyze_tree_pair(
                                to_test, TreeUtils.flip_tree(to_test), verbose=False, plot=False
                            )

                        self.visualizer.print_progress_bar(num_tested, num_trees)

                        if test and gamma is not None:


                            if gamma < best_gamma:
                                best_gamma = gamma
                                best_trees = [sorted(to_test), sorted(TreeUtils.flip_tree(to_test))]

                            if gamma <= 4 / 9:
                                four_nine_gamma_counter += 1
                                four_nine_list.append(to_test)

                            if gamma < 4 / 9:
                                print("\n✅✅✅FOUND BETTER GAMMA✅✅✅", flush=True)
                                print(f"Gamma = {gamma}", flush=True)
                                print(f"Used {sorted(to_test)} and flip", flush=True)
                                exit()
                return

            for i in range(len(points)):
                for j in range(i + 2, len(points)):
                    a, b = points[i], points[j]
                    if (a + 1) % n == b or (b + 1) % n == a:
                        continue

                    edge = (min(a, b), max(a, b))
                    if edge in all_edges:
                        continue

                    new_edge = (a, b)
                    new_local_edges = local_edges + [new_edge]
                    new_all_edges = all_edges | {edge}

                    if has_cycle_uf(new_local_edges):
                        continue

                    a_idx, b_idx = sorted((points.index(a), points.index(b)))
                    between = points[a_idx + 1:b_idx]
                    outside = points[:a_idx + 1] + points[b_idx:]

                    between_edges = [e for e in new_local_edges if e[0] in between and e[1] in between]
                    outside_edges = [e for e in new_local_edges if e[0] in outside and e[1] in outside]

                    enumerate_ncsts_helper(self, between, between_edges, new_all_edges)
                    enumerate_ncsts_helper(self, outside, outside_edges, new_all_edges)

        for necklace in NecklaceGenerator.generate_binary_necklaces(n, k, reflective=False):
            borders = [(i, (i + 1) % n) for i, b in enumerate(necklace) if b == 1]
            border_set = {tuple(sorted(e)) for e in borders}
            enumerate_ncsts_helper(self, list(range(n)), borders, border_set)

        print(f"\nTesting complete. Total NCSTSs tested: {num_tested}")
        print(f"Best gamma: {best_gamma}")
        print(f"Found on trees: {best_trees[0]} and {best_trees[1]}")
        print(f"Number of 4/9 or better: {four_nine_gamma_counter}")
        print(f"All 4/9 or better: {four_nine_list}")
