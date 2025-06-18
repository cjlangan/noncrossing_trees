from ..analysis import GammaAnalyzer
from ..generation import NCSTGenerator

def random_trees_demo():
    gamma_analyzer = GammaAnalyzer()

    print("=== Retrieving a random pair of trees on 15 vertices and analyzing them ===")
    T_i, seed_i = NCSTGenerator.generate_random_ncst(15)
    T_f, seed_f = NCSTGenerator.generate_random_ncst(15)
    gamma_analyzer.analyze_tree_pair(T_i, T_f)
    print(f"Seeds used: {seed_i}, {seed_f}")


if __name__ == "__main__":
    random_trees_demo()
