from ..core import MathUtils


class NCSTFormulas:
    """Class containing formulas for NCST counting."""

    @staticmethod
    def U(n: int, k: int) -> float:
        """
        Compute U(n, k) = (2/(n-2)) * C(n-2, k) * sum_{j=0}^{k-1} C(n-1, j) * C(n-k-2, k-1-j) * 2^{n-1-2k+j}
        """
        if k < 0 or n <= 2 or k >= n:
            return 0

        coeff = 2 / (n - 2)
        binom_term = MathUtils.binomial(n - 2, k)

        sum_value = 0
        for j in range(k):
            term1 = MathUtils.binomial(n - 1, j)
            term2 = MathUtils.binomial(n - k - 2, k - 1 - j)
            exp_term = 2**(n - 1 - 2*k + j)
            sum_value += term1 * term2 * exp_term

        return coeff * binom_term * sum_value

    @staticmethod
    def T(n: int, k: int) -> int:
        """
        Compute T(n, k) = U(n, k-1) - U(n, k) + C(n-1, k) * (1/(n-1)) * sum_{j=0}^{k-1} C(n-1, j) * C(n-k-1, k-1-j) * 2^{n-2k+j}
        """
        if k < 2 or k >= n or n <= 2:
            return 0

        u_diff = NCSTFormulas.U(n, k - 1) - NCSTFormulas.U(n, k)

        coeff = MathUtils.binomial(n - 1, k) / (n - 1)
        sum_value = 0
        for j in range(k):
            term1 = MathUtils.binomial(n - 1, j)
            term2 = MathUtils.binomial(n - k - 1, k - 1 - j)
            exp_term = 2**(n - 2*k + j)
            sum_value += term1 * term2 * exp_term

        additional_term = coeff * sum_value
        return int(u_diff + additional_term)
