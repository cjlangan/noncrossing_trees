import random
import secrets
import numpy as np
import networkx as nx
from conflict import ConflictAnalyzer
from typing import List, Tuple, Optional, Any

from ..core import TreeUtils


class NCSTGenerator:
    """Generator for Non-Crossing Spanning Trees."""

    @staticmethod
    def generate_random_ncst(
        n: int, seed: Optional[int] = None
    ) -> Tuple[List[Tuple[int, int]], int]:
        """Generate a random NCST on n vertices."""
        if seed is None:
            seed = secrets.randbits(32)

        random.seed(seed)
        np.random.seed(seed)

        parent = list(range(n))

        def find(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            return u

        def union(u, v):
            parent[find(u)] = find(v)

        edges = []
        all_possible_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_possible_edges)

        for u, v in all_possible_edges:
            if find(u) != find(v):
                u, v = sorted((u, v))
                valid = True

                # Check for crossings
                for a, b in edges:
                    a, b = sorted((a, b))
                    if len({u, v, a, b}) == 4:
                        if (u < a < v < b) or (a < u < b < v):
                            valid = False
                            break

                if valid:
                    edges.append((u, v))
                    union(u, v)

            if len(edges) == n - 1:
                break

        return edges, seed

    @staticmethod
    def generate_ncst_with_k_borders(
        n: int, k: int, max_tries: int = 100000
    ) -> Tuple[List[Tuple[int, int]], int]:
        """Generate NCST with exactly k border edges."""
        for _ in range(max_tries):
            edges, seed = NCSTGenerator.generate_random_ncst(n)
            num_border = TreeUtils.count_border_edges(edges, n)
            if num_border == k:
                return edges, seed
        raise RuntimeError(f"No NCST with exactly {k} border edges found after {max_tries} tries.")

class ConflictGraphLinearRepGenerator:
    '''
    Generator that generates a family of linear representations of non crossing spanning trees 
    that give you the given conflict graph.

    The algorithm works as follows:
    1. Given a conflict graph, we first partition the edges into the three types of conflict edges mentioned in the paper (Type 1, Type 2, Type 3).
    2. We know that each vertex of the conflict graph corresponds to a gap in the linear representation.
    3. We also can enumerate the gaps in the linear representation.
    4. Now, depending on the type of conflict edge, we can place the edges in the linear representation.
    5. If there are missing vertices in the linear representation, we can add them as isolated vertices 
       at the end of the linear representation.
    6. Finally, we return the linear representation of the non crossing spanning tree.
    '''

    def __init__(self, conflict_graph: nx.Graph):
        self.conflict_graph = conflict_graph
        self.conflict_vertices = sorted(list(conflict_graph.nodes()))
        self.conflict_edges = [
            (u, v, data['type']) for u, v, data in conflict_graph.edges(data=True)
        ]
        self.n = len(self.conflict_vertices)
        self.num_gaps = self.n

    def _possible_near_edges(self, idx: int, n: int) -> List[Tuple[int, int]]:
        # Returns all possible near edges for a gap idx in a linear representation of size n+1
        # Near edges are (idx, b) with b > idx+1 and b <= n
        return [(idx, b) for b in range(idx+2, n+1)]

    def _possible_short_edges(self, n: int) -> List[Tuple[int, int]]:
        # Short edges are (i, i+1)
        return [(i, i+1) for i in range(n)]

    def _edges_for_assignment(self, assignment: List[Optional[Tuple[int, int]]], n: int) -> List[Tuple[int, int]]:
        # Returns the full set of edges for a tree: short edges + assigned near edges
        edges = self._possible_short_edges(n)
        for edge in assignment:
            if edge is not None:
                edges.append(edge)
        return edges

    def _check_conflict_graph(
        self,
        T_i_edges: List[Tuple[int, int]],
        T_f_edges: List[Tuple[int, int]]
    ) -> bool:
        conflict_vertices, E_i, E_f = ConflictAnalyzer.get_gaps_and_edge_pairs(T_i_edges, T_f_edges)
        computed_edges = ConflictAnalyzer.get_conflict_edges(conflict_vertices, E_i, E_f)
        computed_set = set((u, v, t) for u, v, t in computed_edges)
        target_set = set((u, v, t) for u, v, t in self.conflict_edges)
        # Check if the computed conflict graph matches the input. This is a necessary condition, but we could probably speed this up or something? This will be slow for large conflict graphs.
        return computed_set == target_set and set(conflict_vertices) == set(self.conflict_vertices)

    def _backtrack(
        self,
        idx: int,
        T_i_assignment: List[Optional[Tuple[int, int]]],
        T_f_assignment: List[Optional[Tuple[int, int]]],
        n: int
    ) -> Optional[Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]]:
        if idx == self.num_gaps:
            # All assignments made, check if conflict graph matches. If it does not, God help us :((((.
            T_i_edges = self._edges_for_assignment(T_i_assignment, n)
            T_f_edges = self._edges_for_assignment(T_f_assignment, n)
            if self._check_conflict_graph(T_i_edges, T_f_edges):
                return (T_i_edges, T_f_edges)
            return None

        # Try all possible near edges (or None) for this gap in both trees
        possible_i = [None] + self._possible_near_edges(idx, n)
        possible_f = [None] + self._possible_near_edges(idx, n)
        for near_i in possible_i:
            for near_f in possible_f:
                # Avoid duplicate edges
                if near_i is not None and near_i in T_i_assignment:
                    continue
                if near_f is not None and near_f in T_f_assignment:
                    continue
                T_i_assignment[idx] = near_i
                T_f_assignment[idx] = near_f
                result = self._backtrack(idx+1, T_i_assignment, T_f_assignment, n)
                if result is not None:
                    return result
                T_i_assignment[idx] = None
                T_f_assignment[idx] = None
        return None

    def generate(self) -> Optional[Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]]:
        '''
        Attempt to generate two linear representations (T_i, T_f)
        whose conflict graph matches the input.
        Returns (T_i, T_f) or None if not possible.
        '''
        n = self.num_gaps
        T_i_assignment: List[Optional[Tuple[int, int]]] = [None] * n
        T_f_assignment: List[Optional[Tuple[int, int]]] = [None] * n
        return self._backtrack(0, T_i_assignment, T_f_assignment, n)