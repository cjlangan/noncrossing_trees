import networkx as nx
import numpy as np
import math
from typing import List, Tuple, Optional
from .data_structures import UnionFind
from itertools import chain


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
    def is_path_graph(graph):
        """
        Check if a graph is a path (linear chain of nodes).
        
        A path graph has these properties:
        1. Exactly 2 nodes with degree 1 (endpoints)
        2. All other nodes have degree 2 (middle nodes)
        """
        if len(graph.nodes()) == 0:
            return False
        
        if len(graph.nodes()) == 1:
            return True  # Single node is trivially a path
        
        if len(graph.nodes()) == 2:
            return len(graph.edges()) == 1  # Two nodes connected by one edge
        
        # Count degrees
        degrees = [graph.degree(n) for n in graph.nodes()]
        degree_counts = {}
        for d in degrees:
            degree_counts[d] = degree_counts.get(d, 0) + 1
        
        # Path graph must have:
        # - Exactly 2 nodes of degree 1 (endpoints)
        # - All other nodes of degree 2 (middle nodes)
        return (degree_counts.get(1, 0) == 2 and 
                degree_counts.get(2, 0) == len(graph.nodes()) - 2)


    @staticmethod
    def cyclic_trim(points, a, b):
        """Trim a list as if it were cyclic"""
        if not points:
            return []
        
        n = len(points)
        a, b = a % n, b % n
        
        if a <= b:
            return points[a:b+1]
        else:
            return list(chain(points[a:], points[:b+1]))
        
    @staticmethod
    def find_edge_from_gap(tree: List[Tuple[int, int]], gap: int) -> Tuple[int, int]:
        """Find the edge associated with a specific gap."""
        min_distance = len(tree) + 1
        res_u = res_v = -1
        for u, v in tree:
            if u <= gap < v or v <= gap < u:
                if (abs(u - v) < min_distance):
                    min_distance = abs(u - v)
                    res_u, res_v = u, v
        return res_u, res_v

    @staticmethod
    def is_near_gap(tree: List[Tuple[int, int]], gap: int) -> bool:
        """Check if a gap is a near edge."""
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

    @staticmethod
    def reduce_gap(tree: List[Tuple[int, int]], gap: int) -> List[Tuple[int, int]]:
        """Reduce tree by removing edges associated with a specific gap."""
        return [(a, b) if (a, b) != TreeUtils.find_edge_from_gap(tree, gap) else (gap, gap + 1) for a, b in tree]

    @staticmethod
    def reduce_tree_pair(tree_i: List[Tuple[int, int]], tree_f: List[Tuple[int, int]], verbose: Optional[bool] = True) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Reduce a tree pair by removing redundant gaps"""

        reduced_i = [sorted((a, b)) for a, b in tree_i]
        reduced_f = [sorted((a, b)) for a, b in tree_f]
        gaps_i = [0] * len(tree_i)
        gaps_f = [0] * len(tree_f)

        for i in range(len(tree_i)):
            u, v = TreeUtils.find_edge_from_gap(tree_i, i)
            for j in range(len(tree_i)):
                if tree_i[j][0] == u and tree_i[j][1] == v:
                    gaps_i[j] = i
                    break

        for i in range(len(tree_f)):
            u, v = TreeUtils.find_edge_from_gap(tree_f, i)
            for j in range(len(tree_f)):
                if tree_f[j][0] == u and tree_f[j][1] == v:
                    gaps_f[j] = i
                    break

        bad_gaps = [i for i in range(len(tree_i)) if not TreeUtils.is_near_near_gap(tree_i, tree_f, i)][::-1]

        for gap in bad_gaps:
            reduced_i = TreeUtils.reduce_gap(reduced_i, gap)
            reduced_f = TreeUtils.reduce_gap(reduced_f, gap)

        for gap in bad_gaps:
            # Check if gap can be removed: It can be removed if there doesn't exist two edges not associated to the gap such that they only overlap at the gap
            u_i, v_i = TreeUtils.find_edge_from_gap(reduced_i, gap)
            u_f, v_f = TreeUtils.find_edge_from_gap(reduced_f, gap)
            if (TreeUtils.is_near_near_gap(reduced_i, reduced_f, gap - 1) and (TreeUtils.find_edge_from_gap(reduced_i, gap - 1)[1] == gap + 1 or TreeUtils.find_edge_from_gap(reduced_f, gap - 1)[1] == gap + 1)):
                continue            
            if (TreeUtils.is_near_near_gap(reduced_i, reduced_f, gap + 1) and (TreeUtils.find_edge_from_gap(reduced_i, gap + 1)[0] == gap or TreeUtils.find_edge_from_gap(reduced_f, gap + 1)[0] == gap)):
                continue
            if not any([(a, b) != (u_i, v_i) and
                        (c, d) != (u_f, v_f) and
                        (max(a, c) == gap) and
                        (min(b, d) == gap + 1)
                        for a, b in reduced_i for c, d in reduced_f]):
                if verbose: 
                    print(f"Removing gap {gap} from trees")
                reduced_i = TreeUtils.remove_gap(reduced_i, gap)
                reduced_f = TreeUtils.remove_gap(reduced_f, gap)
                gaps_i = [g - 1 if g > gap else g for g in gaps_i if g != gap]
                gaps_f = [g - 1 if g > gap else g for g in gaps_f if g != gap]

        if verbose:
            print("Reduction complete.")

        return reduced_i, reduced_f


    @staticmethod
    def cross(e1: Tuple[int, int], e2: Tuple[int, int]) -> bool:
        """Determine if two edges cross"""

        a,b = sorted(e1)
        c,d = sorted(e2)

        return a < c < b < d or c < a < d < b

    @staticmethod
    def is_short(edge: Tuple[int, int]) -> bool:
        """Determine if an edge is short"""
        a,b = edge
        return abs(b - a) == 1

    @staticmethod
    def edge_length(edge: Tuple[int, int], n: int) -> int:
        """Determines legth of edge on a convex polygon"""
        max_length = math.floor(n/2)
        a,b = edge
        length = linear_length = abs(b - a)

        if linear_length > max_length:
            length = n - linear_length

        return length
