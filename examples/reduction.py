from ..core import TreeUtils
from ..generation import NCSTGenerator
from ..analysis import GammaAnalyzer

def reduction_demo():
    gen = NCSTGenerator()
    ana = GammaAnalyzer()

    print("Reducing 2 random trees...")

    T_i = gen.generate_ncst_with_k_borders(30, 4)[0]
    T_f = gen.generate_ncst_with_k_borders(30, 4)[0]

    reduced_T_i, reduced_T_f = TreeUtils.reduce_tree_pair(T_i, T_f, verbose=True)

    gamma = ana.analyze_tree_pair(T_i, T_f, plot=False)[0]
    gamma_reduced = ana.analyze_tree_pair(reduced_T_i, reduced_T_f)[0]

    print(f"Original gamma: {gamma}\nReduced gamma: {gamma_reduced}")

    if gamma != gamma_reduced:
        print("ERROR: GAMMAS ARE DIFFERENT")


if __name__ == "__main__":
    reduction_demo()
