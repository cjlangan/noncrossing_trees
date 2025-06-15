from typing import List


class MathUtils:
    """Utility class for mathematical computations."""

    @staticmethod
    def binomial(n: int, k: int) -> int:
        """Compute binomial coefficient C(n, k) = n! / (k! * (n-k)!)"""
        if k < 0 or k > n:
            return 0
        if k == 0 or k == n:
            return 1

        result = 1
        for i in range(min(k, n - k)):
            result = result * (n - i) // (i + 1)
        return result

    @staticmethod
    def totient(n: int) -> int:
        """Euler's totient function."""
        result = n
        p = 2
        while p * p <= n:
            if n % p == 0:
                while n % p == 0:
                    n //= p
                result -= result // p
            p += 1
        if n > 1:
            result -= result // n
        return result

    @staticmethod
    def divisors(n: int) -> List[int]:
        """Get all divisors of n."""
        divs = []
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                divs.append(i)
                if i != n // i:
                    divs.append(n // i)
        return sorted(divs)
