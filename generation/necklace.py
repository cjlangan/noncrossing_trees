from typing import List, Tuple
from more_itertools import distinct_permutations
from math import gcd, comb

from ..core import MathUtils


class NecklaceGenerator:
    """Generator for binary necklaces (border combinations)."""

    @staticmethod
    def count_binary_necklaces(n: int, k: int) -> int:
        """
        Calculate the number of binary necklaces
        of length n with exactly k ones.
        """
        if k < 0 or k > n:
            return 0

        g = gcd(n, k)
        total = 0

        for d in MathUtils.divisors(g):
            phi_d = MathUtils.totient(d)
            binom = comb(n // d, k // d)
            total += phi_d * binom

        return int(total // n)

    @staticmethod
    def rotate(lst: List[int], k: int) -> List[int]:
        """Rotate list by k positions."""
        return lst[k:] + lst[:k]

    @staticmethod
    def min_rotation(bitstring: List[int]) -> Tuple[int, ...]:
        """Find lexicographically smallest rotation."""
        return min(tuple(NecklaceGenerator.rotate(bitstring, i))
                   for i in range(len(bitstring)))

    @staticmethod
    def generate_binary_necklaces(n: int, k: int, reflective: bool = False):
        """Generate binary necklaces of length n with k ones."""
        seen = set()
        base = [1] * k + [0] * (n - k)

        for perm in distinct_permutations(base):
            canon = NecklaceGenerator.min_rotation(list(perm))

            if reflective:
                flipped = NecklaceGenerator.min_rotation(list(perm[::-1]))
                canon = min(canon, flipped)

            if canon not in seen:
                seen.add(canon)
                yield canon
