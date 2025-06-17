from ..analysis import GammaAnalyzer

def blowup_demo():
    gamma_analyzer = GammaAnalyzer()

    # This is the best pair of trees (lowest known gamma with the lowest number of vertices)
    T_i = [(5, 9), (0, 9), (1, 4), (0, 11), (1, 5), (7, 8),
           (2, 4), (3, 4), (6, 8), (5, 8), (0, 10)]
    T_f = [(6, 2), (11, 2), (10, 7), (11, 0), (10, 6),
           (4, 3), (9, 7), (8, 7), (5, 3), (6, 3), (11, 1)]

    # Analyze the blowup of the best pair, with one edge insertion per gap
    print("=== Analyzing the best known trees with a k-blowup of 1 ===")
    gamma_analyzer.analyze_pair_blowup(T_i, T_f, 1)

if __name__ == "__main__":
    blowup_demo()
