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
    def find_edge_from_gap(tree: List[Tuple[int, int]], gap: int) -> Tuple[int, int]:
        """Find the edge associated with a specific gap."""
        min_distance = len(tree) + 1
        res_u = res_v = None
        for u, v in tree:
            if u <= gap < v or v <= gap < u:
                if (abs(u - v) < min_distance):
                    min_distance = abs(u - v)
                    res_u, res_v = u, v
        return res_u, res_v
    
    @staticmethod
    def is_near_gap(tree: List[Tuple[int, int]], gap: int) -> bool:
        """Check if a gap is a near pair."""
        u, v = TreeUtils.find_edge_from_gap(tree, gap)
        return (u == gap or u == gap + 1) ^ (v == gap or v == gap + 1)
    
    @staticmethod
    def is_near_near_gap(tree: List[Tuple[int, int]], tree2: List[Tuple[int, int]], gap: int) -> bool:
        """Check if a gap is a near-near pair."""
        u, v = TreeUtils.find_edge_from_gap(tree, gap)
        u2, v2 = TreeUtils.find_edge_from_gap(tree2, gap)
        return TreeUtils.is_near_gap(tree, gap) and TreeUtils.is_near_gap(tree2, gap)

    @staticmethod
    def remove_gap(tree: List[Tuple[int, int]], gap: int) -> List[Tuple[int, int]]:
        """Remove edges associated with a specific gap."""
        return [(a - 1 if a > gap else a,
                 b - 1 if b > gap else b) for a, b in tree if (a, b) != TreeUtils.find_edge_from_gap(tree, gap)]
    
    def reduce_gap(tree: List[Tuple[int, int]], gap: int) -> List[Tuple[int, int]]:
        """Reduce tree by removing edges associated with a specific gap."""
        return [(a, b) if (a, b) != TreeUtils.find_edge_from_gap(tree, gap) else (gap, gap + 1) for a, b in tree]
        
    @staticmethod
    def reduce_tree_pair(tree_i: List[Tuple[int, int]], tree_f: List[Tuple[int, int]]) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Reduce a tree pair by removing redundant gaps"""
        reduced_i = tree_i[:]
        reduced_f = tree_f[:]
        bad_gaps = [i for i in range(len(tree_i)) if not TreeUtils.is_near_near_gap(tree_i, tree_f, i)][::-1]
        for gap in bad_gaps:
            # Check if gap can be removed: It can be removed if there doesn't exist two edges not associated to the gap such that they only overlap at the gap
            u_i, v_i = TreeUtils.find_edge_from_gap(tree_i, gap)
            u_f, v_f = TreeUtils.find_edge_from_gap(tree_f, gap)
            if not any([(a, b) != (u_i, v_i) and
                        (c, d) != (u_f, v_f) and
                        (a == gap + 1 or b == gap + 1) and
                        (c == gap or d == gap) 
                        for a, b in reduced_i for c, d in reduced_f]):
                reduced_i = TreeUtils.remove_gap(reduced_i, gap)
                reduced_f = TreeUtils.remove_gap(reduced_f, gap)
            else:
                reduced_i = TreeUtils.reduce_gap(reduced_i, gap)
                reduced_f = TreeUtils.reduce_gap(reduced_f, gap)
            
        return reduced_i, reduced_f