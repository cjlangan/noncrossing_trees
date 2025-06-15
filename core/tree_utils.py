import networkx as nx
from typing import List, Tuple
from .data_structures import UnionFind


class TreeUtils:
    """Utility class for tree operations and transformations."""

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
        flipped = [((n - a) % n, (n - b) % n) for a, b in tree]
        return TreeUtils.rotate_tree(flipped, -1)

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
