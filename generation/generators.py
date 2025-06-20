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


    # WORK IN PROGRESS:
    # NEED TO CONSIDER ALL POSSIBLE CHORDS, SINCE YOU CAN HIT IMPOSSIBLE CASES
    @staticmethod
    def generate_ncst_with_k_borders( n: int, k: int, seed: Optional[int] = None) -> Tuple[List[Tuple[int, int]], int]:
        """Generate random NCST with exactly k border edges by recursive approach"""
        if seed is None:
            seed = secrets.randbits(32)

        random.seed(seed)
        np.random.seed(seed)

        # Get all borders that aren't (0, n-1)
        all_borders = [(i, i+1) for i in range(0, n-1)]

        # Choose k-1 random borders that arent (0, n-1)
        # We forcefully include (0, n-1)
        borders = [(0, n-1)] + random.sample(all_borders, k-1)

        print("Borders:", borders)

        # Add border to all edges
        edges = borders.copy()

        # Helper function for recursivley choosing all chords randomly
        def choose_chords(points: List[int], local_edges: List[Tuple[int, int]]):
            nonlocal edges

            # Base case: have a tree
            if len(local_edges) == len(points) - 1:
                return

            # Recursive case: Choose a chord and split into subproblems

            # First choose chord start point
            P = len(points) # number of local points
            start_idx = random.randrange(P)
            left_idx = start_idx - 1 if start_idx > 0 else P - 1

            # Remove that point and its neighbors from possible options
            leftover_indicies = list(range(P))
            leftover_indicies.remove(start_idx)
            leftover_indicies.remove(left_idx)
            leftover_indicies.remove((start_idx + 1) % P)

            # Choose chord endpoint
            end_idx = random.choice(leftover_indicies)

            # Define new chord edge and add it to edges and local edges
            chord = (points[start_idx], points[end_idx])
            print("Chord chosen:", chord)
            edges = edges + [chord]
            new_local_edges = local_edges + [chord]

            # Sort chosen indicies to partition for subproblems
            a_idx, b_idx = sorted((start_idx, end_idx))

            # Define points between and outside of chosen chord
            between = points[a_idx + 1:b_idx]
            outside = points[:a_idx + 1] + points[b_idx:]

            # Determine edges in point range for subproblems
            between_edges = [e for e in new_local_edges if e[0] in between and e[1] in between]
            outside_edges = [e for e in new_local_edges if e[0] in outside and e[1] in outside]

            # Recursively choose chord for between and outside subproblem
            choose_chords(between, between_edges)
            choose_chords(outside, outside_edges)

        # Choose all chords based off initial borders and point list
        choose_chords(list(range(n)), borders)

        return edges, seed

