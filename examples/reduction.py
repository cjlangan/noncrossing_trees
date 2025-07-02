from ..core import TreeUtils
from ..generation import NCSTGenerator
from ..analysis import GammaAnalyzer
import time

def reduction_demo():
    gen = NCSTGenerator()
    ana = GammaAnalyzer()

    T_i = gen.generate_ncst_with_k_borders(30, 4)[0]
    T_f = gen.generate_ncst_with_k_borders(30, 4)[0]

    # T_i = [(8, 14), (3, 15), (9, 11), (3, 8), (9, 13), (0, 15), (5, 8), (5, 7), (4, 8), (2, 15), (9, 12), (1, 15), (10, 11), (13, 14), (5, 6)]
    # T_f = [(7, 1), (12, 0), (6, 4), (12, 7), (6, 2), (15, 0), (10, 7), (10, 8), (11, 7), (13, 0), (6, 3), (14, 0), (5, 4), (2, 1), (10, 9)]

    T_i = [(0, 29), (10, 11), (18, 19), (4, 5), (11, 24), (13, 19), (17, 19), (14, 19), (16, 19), (15, 19), (11, 19), (11, 23), (11, 20), (11, 22), (11, 21), (12, 19), (2, 26), (0, 28), (0, 26), (1, 26), (0, 27), (3, 11), (5, 11), (9, 11), (8, 11), (6, 11), (7, 11), (3, 26), (11, 25)]
    T_f = [(0, 29), (20, 21), (2, 3), (3, 4), (1, 24), (7, 21), (10, 21), (18, 21), (19, 21), (15, 21), (14, 21), (11, 21), (13, 21), (12, 21), (17, 21), (16, 21), (9, 21), (8, 21), (2, 5), (7, 23), (7, 22), (1, 6), (1, 5), (1, 23), (1, 29), (1, 28), (1, 26), (1, 25), (1, 27)]


    reduced_T_i, reduced_T_f = TreeUtils.reduce_tree_pair(T_i, T_f)

    gamma = ana.analyze_tree_pair(T_i, T_f)[0]
    gamma_reduced = ana.analyze_tree_pair(reduced_T_i, reduced_T_f)[0]

    print("Original Trees:")
    print(T_i)
    print(T_f)
    print("Reduced Trees:")
    print(reduced_T_i)
    print(reduced_T_f)

    print(gamma, gamma_reduced)


if __name__ == "__main__":
    reduction_demo()
