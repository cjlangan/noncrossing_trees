from ..analysis import Greedy
from ..generation import NCSTGenerator
from ..visualization import Visualizer

def test_demo():
    greed = Greedy()
    gen = NCSTGenerator()

    T_i = gen.generate_random_ncst(20)[0]
    T_f = gen.generate_random_ncst(20)[0]

    # Visualizer.plot_trees_together(T_i, T_f, "original_trees.png")
    # Visualizer.plot_linear_graph(T_i, T_f, T_i, T_f, "linear_graph.png")

    greed.get_flip_sequence(T_i, T_f)


if __name__ == "__main__":
    test_demo()
