from ..analysis import GammaAnalyzer

def best_trees_demo():
    gamma_analyzer = GammaAnalyzer()
        
    # This is the best pair of trees (lowest known gamma with the lowest number of vertices)
    T_i = [(5, 9), (0, 9), (1, 4), (0, 11), (1, 5), (7, 8),
           (2, 4), (3, 4), (6, 8), (5, 8), (0, 10)]
    T_f = [(6, 2), (11, 2), (10, 7), (11, 0), (10, 6),
           (4, 3), (9, 7), (8, 7), (5, 3), (6, 3), (11, 1)]

    print("=== Analyzing Best Known Tree Pair (gamma ≈ 0.44444444, n=12) ===")
    gamma_analyzer.analyze_tree_pair(T_i, T_f, verbose=True, plot=True)

if __name__ == "__main__":
    best_trees_demo()
