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
    def generate_ncst_with_k_borders(
        n: int, k: int, max_tries: int = 100000
    ) -> Tuple[List[Tuple[int, int]], int]:
        """Generate NCST with exactly k border edges."""
        for _ in range(max_tries):
            edges, seed = NCSTGenerator.generate_random_ncst(n)
            num_border = TreeUtils.count_border_edges(edges, n)
            if num_border == k:
                return edges, seed
        raise RuntimeError(f"No NCST with exactly {
                           k} border edges found after {max_tries} tries.")
