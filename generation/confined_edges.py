from typing import List, Tuple

class ConfinedEdgeGenerator:

    @staticmethod
    def evenly_spaced_border_combination(n: int, k: int) -> List[Tuple[int, int]]:
        """Generate k evenly spaced unique borders on n points, including (n-1, 0)"""
        if k == 0:
            return []
        #if n % k != 0:
        #    raise ValueError(f"k = {k} must divide n = {n} evenly")

        step = n // k
        # Shift all indices back by 1 to include (n-1, 0)
        border_edges = [((i * step - 1) % n, ((i * step) % n)) for i in range(k)]
        return border_edges
