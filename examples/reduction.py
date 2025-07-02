from ..core import TreeUtils
from ..generation import NCSTGenerator
from ..analysis import GammaAnalyzer
import time

def reduction_demo():
    gen = NCSTGenerator()
    ana = GammaAnalyzer()

    T_i = gen.generate_ncst_with_k_borders(30, 4)[0]
    T_f = gen.generate_ncst_with_k_borders(30, 4)[0]

    reduced_T_i, reduced_T_f = TreeUtils.reduce_tree_pair(T_i, T_f)

    gamma = ana.analyze_tree_pair(T_i, T_f)[0]

    time.sleep(15)

    gamma_reduced = ana.analyze_tree_pair(reduced_T_i, reduced_T_f)[0]

    print(gamma, gamma_reduced)


if __name__ == "__main__":
    reduction_demo()
