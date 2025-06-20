import networkx as nx
import numpy as np
from typing import List, Tuple
from .data_structures import UnionFind


class TreeUtils:
    """Utility class for tree operations and transformations."""

    @staticmethod
    def blowup_tree(T: List[Tuple[int, int]], k: int, E: List[Tuple[int, int]], conflict_vertices: List[int]) -> List[Tuple[int, int]]:
        """
        Blows up a tree by inserting k edges bewteen each gap

        PARAMETERS:
            T: the tree
            k: number of added edges per gap
            E: T's near edges from near-near pairs
            conflict_vertices: the conflict graphs conflict vertices
        """
        T_blown = list(T)
        all_vertices = set()

        for i in range(len(E)):
            a, b = sorted(E[i])
            cv = conflict_vertices[i] # the conflict vertex

            # Evenly space new vertices within the gap
            new_fractions = np.linspace(cv, cv+1, k + 2)[1:-1]
            all_vertices.update(new_fractions)
                
            # Determine endpoint
            endpoint = b if a == cv else a

            # Add new edges to tree
            for v in new_fractions:
                T_blown.append((v, endpoint))

        # Normalize all vertices: assign integer labels based on sorted order
        for u, v in T_blown:
            all_vertices.add(u)
            all_vertices.add(v)

        sorted_vertices = sorted(all_vertices)
        mapping = {v: i for i, v in enumerate(sorted_vertices)}

        def normalize(v):
            return mapping[v]

        T_normalized = [(normalize(u), normalize(v)) for (u, v) in T_blown]

        return T_normalized

    @staticmethod
    def is_valid_tree(vertices: List[int], edges: List[Tuple[int, int]]) -> bool:
        """Check if the given edges form a valid tree on the vertices."""
        if len(edges) != len(vertices) - 1:
            return False
        G = nx.Graph()
        G.add_nodes_from(vertices)
        G.add_edges_from(edges)
        return nx.is_connected(G)

    @staticmethod
    def has_cycle(edges: List[Tuple[int, int]]) -> bool:
        """Check if edges contain a cycle using Union-Find."""
        uf = UnionFind()
        for u, v in edges:
            if uf[u] == uf[v]:
                return True
            uf.union(u, v)
        return False

    @staticmethod
    def rotate_tree(tree: List[Tuple[int, int]], k: int) -> List[Tuple[int, int]]:
        """Rotate tree by k positions."""
        n = len(tree) + 1
        return [((a + k) % n, (b + k) % n) for a, b in tree]

    @staticmethod
    def flip_tree(tree: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Flip tree and normalize."""
        n = len(tree) + 1
        flipped = [((n - a - 1) % n, (n - b - 1) % n) for a, b in tree]
        return flipped

    @staticmethod
    def normalize_edges(edges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Normalize edges to have smaller vertex first."""
        return [(min(a, b), max(a, b)) for a, b in edges]

    @staticmethod
    def is_hull_edge(u: int, v: int, n: int) -> bool:
        """Check if edge is on the convex hull."""
        return (abs(u - v) == 1) or (u == 0 and v == n - 1) or (u == n - 1 and v == 0)

    @staticmethod
    def count_border_edges(edges: List[Tuple[int, int]], n: int) -> int:
        """Count number of border (hull) edges."""
        return sum(1 for u, v in edges if TreeUtils.is_hull_edge(u, v, n))
    
    @staticmethod
    def get_border_edges(tree: List[Tuple[int, int]], n: int) -> List[Tuple[int, int]]:
        """Get the border edges of a tree."""
        return set((min(u, v), max(u, v)) for u, v in tree if TreeUtils.is_hull_edge(u, v, n))

    @staticmethod
    def trees_share_borders(t1: List[Tuple[int, int]], t2: List[Tuple[int, int]]) -> bool:
        # Check if two trees have the same border edges and the other edges are distinct
        border1 = TreeUtils.get_border_edges(t1, len(t1) + 1)
        border2 = TreeUtils.get_border_edges(t2, len(t2) + 1)
        if border1 != border2:
            return False
        edges1 = set(TreeUtils.normalize_edges(t1))
        edges2 = set(TreeUtils.normalize_edges(t2))
        # Remove border edges from both sets
        edges1 -= border1
        edges2 -= border2
        # Check if the remaining edges are distinct
        return edges1.isdisjoint(edges2)
