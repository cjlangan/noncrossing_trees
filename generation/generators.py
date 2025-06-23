import networkx as nx
import random
import secrets
import numpy as np
from typing import List, Tuple, Optional

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
    def generate_ncst_with_k_borders_old(
        n: int, k: int, max_tries: int = 100000
    ) -> Tuple[List[Tuple[int, int]], int]:
        """Generate NCST with exactly k border edges."""
        for _ in range(max_tries):
            edges, seed = NCSTGenerator.generate_random_ncst(n)
            num_border = TreeUtils.count_border_edges(edges, n)
            if num_border == k:
                return edges, seed
        raise RuntimeError(f"No NCST with exactly {k} border edges found after {max_tries} tries.")


    @staticmethod
    def generate_ncst_with_k_borders( n: int, k: Optional[int] = None, seed: Optional[int] = None, 
                                     given_borders: Optional[List[Tuple[int, int]]] = None) -> Tuple[List[Tuple[int, int]], int]:
        """Generate random NCST with exactly k border edges by recursive deterministic approach"""
        if seed is None:
            seed = secrets.randbits(32)

        random.seed(seed)
        np.random.seed(seed)
        borders = given_borders if given_borders is not None else []

        # If no given borders, randomly choose them
        if given_borders is None and k is not None:
            # Get all borders that aren't (0, n-1)
            all_borders = [(i, i+1) for i in range(0, n-1)]

            # Choose k-1 random borders that arent (0, n-1)
            # We forcefully include (0, n-1)
            borders = [(0, n-1)] + random.sample(all_borders, k-1)

        edges = borders.copy()

        # Helper function for recursivley choosing all chords randomly
        def choose_chords(points: List[int], local_edges: List[Tuple[int, int]]):
            nonlocal edges


            # Base case: have a tree
            if len(local_edges) == len(points) - 1:
                return

            graph = nx.Graph()
            graph.add_edges_from(local_edges)
            P = len(points) # number of local points

            initial_points = points.copy()

            # leaves of the only path are invalid, so remove 
            if TreeUtils.is_path_graph(graph):
                leaves = [x for x in graph.nodes() if graph.degree(x) == 1]
                initial_points = [item for item in points if item not in leaves]

            # Choose starting point from valid points
            start_point = random.choice(initial_points)
            start_idx = points.index(start_point)

            # Function to find first valid index for chord endpoint in specific direction
            def find_valid_endpoint_idx(start_idx: int, dir: int) -> int:
                # Init first edge
                a = start_idx
                b = (a + dir) % P

                border = False
                non_border = False

                # Check potential border edges until found both a border edge 
                # and a non border edge
                while not border or not non_border:
                    p1 = min(points[a], points[b])
                    p2 = max(points[a], points[b])

                    if (p1, p2) in local_edges: # THIS ASSUMES EDGE IS SORTED -> BAD
                        border = True
                    else:
                        non_border = True

                    # Move to next potential border edge
                    a = b
                    b = (a + dir) % P

                # 'a' ends up being the first valid index
                return a


            # Find counter clockwise and clockwise beginning chord index ends
            cc_idx = find_valid_endpoint_idx(start_idx, 1)
            c_idx = find_valid_endpoint_idx(start_idx, -1)

            # Create list of all valid endpoints between these points
            valid_endpoints = TreeUtils.cyclic_trim(points, cc_idx, c_idx)

            end_point = random.choice(valid_endpoints)
            end_idx = points.index(end_point)

            # Define new chord edge and add it to edges and local edges
            chord = (min(start_point, end_point), max(start_point, end_point))
            edges = edges + [chord]
            new_local_edges = local_edges + [chord]

            # Sort chosen indicies to partition for subproblems
            a_idx, b_idx = sorted((start_idx, end_idx))

            # Define points between and outside of chosen chord
            between = TreeUtils.cyclic_trim(points, a_idx, b_idx)
            outside = TreeUtils.cyclic_trim(points, b_idx, a_idx)

            # Determine edges in point range for subproblems
            between_edges = [e for e in new_local_edges if e[0] in between and e[1] in between]
            outside_edges = [e for e in new_local_edges if e[0] in outside and e[1] in outside]

            # Recursively choose chord for between and outside subproblem
            choose_chords(between, between_edges)
            choose_chords(outside, outside_edges)

        # Choose all chords based off initial borders and point list
        choose_chords(list(range(n)), borders)

        return edges, seed

