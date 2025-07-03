from ..core import TreeUtils
from ..generation import NCSTGenerator
from ..analysis import GammaAnalyzer

def reduction_demo():
    gen = NCSTGenerator()
    ana = GammaAnalyzer()

    n = 30
    k = 4
    iterations = 10

    print(f"=== Testing {iterations} random NCSTs on {n} vertices and {k} ===")
    print(f"=== borders against their reduction counterpart. ===")

    for i in range(iterations):
        T_i, si = gen.generate_ncst_with_k_borders(n, k)
        T_f, sf = gen.generate_ncst_with_k_borders(n, k)

        reduced_T_i, reduced_T_f = TreeUtils.reduce_tree_pair(T_i, T_f, verbose=False)

        gamma = ana.analyze_tree_pair(T_i, T_f, verbose=False, plot=False)[0]
        gamma_reduced = ana.analyze_tree_pair(reduced_T_i, reduced_T_f, verbose=False, plot=False)[0]

        if gamma_reduced == gamma:
            print(f"Test {i}: SAME GAMMAS")
        else:
            print(f"Test {i}: xxxxxxxxxxxx DIFFERENT GAMMAS xxxxxxxxxxxx")
            print(f"Seeds: {si} and {sf}")
            print(f"Trees: {T_i} and {T_f}")

if __name__ == "__main__":
    reduction_demo()
