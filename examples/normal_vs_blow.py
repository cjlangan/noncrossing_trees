from ..generation import NCSTGenerator
from ..analysis import GammaAnalyzer


def normal_vs_blowup_demo():
    gen = NCSTGenerator()
    ana = GammaAnalyzer()

    n = 20
    k = 4
    k_blow = 2
    iterations = 30

    print(f"=== Testing {iterations} random NCSTs on {n} vertices with {k} ===")
    print(f"=== borders against their {k_blow}-blowup counterpart.   ===")

    for i in range(10):
        T_i, si = gen.generate_ncst_with_k_borders(n, k)
        T_f, sf = gen.generate_ncst_with_k_borders(n, k)
        gamma_norm = ana.analyze_tree_pair(T_i, T_f, verbose=False, plot=False)[0]
        gamma_blow = ana.analyze_pair_blowup(T_i, T_f, k_blow, verbose=False, plot=False)[0]

        if gamma_blow == gamma_norm:
            print(f"Test {i}: SAME GAMMAS")
        else:
            print(f"Test {i}: xxxxxxxxxxxx DIFFERENT GAMMAS xxxxxxxxxxxx")
            print(f"Seeds: {si} and {sf}")
            print(f"Trees: {T_i} and {T_f}")


if __name__ == "__main__":
    normal_vs_blowup_demo()
