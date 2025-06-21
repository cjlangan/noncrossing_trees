from typing import List, Tuple

class ConfinedEdgeGenerator:

    @staticmethod
    def evenly_spaced_border_combination(n: int, k: int) -> List[Tuple[int, int]]:
        """Generate k evenly spaced unique borders on n points, including (0, n-1)"""
        if k == 0:
            return []
        if n % k != 0:
            raise ValueError(f"k = {k} must divide n = {n} evenly")

        step = n // k
        border_edges = [((i * step - 1) % n, ((i * step) % n)) for i in range(k)]
        border_edges[0] = (0, n-1)
        return border_edges
