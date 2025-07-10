from ..analysis import Greedy
from ..generation import NCSTGenerator
from ..visualization import Visualizer

def test_demo():
    greed = Greedy()
    gen = NCSTGenerator()

    # T_i = gen.generate_random_ncst(20)[0]
    # T_f = gen.generate_random_ncst(20)[0]

    # Symmetric n=13 pair
    # T_i = [(0,3),(3,12),(1,3),(4,7),(4,8),(5,7),(8,11),(8,12),(9,11),(2, 3),(6,7),(10,11)]
    # T_f = [(0,10),(1,10),(1,5),(2,4),(2,5),(5,9),(6,8),(6,9),(10,12),(2,3),(6,7),(10,11)]

    # Paper trees
    T_i = [(0,3),(0,4),(1,3),(4,7),(4,8),(5,7),(8,11),(8,12),(9,11),(2, 3),(6,7),(10,11)]
    T_f = [(0,10),(1,10),(1,5),(2,4),(2,5),(5,9),(6,8),(6,9),(10,12),(2,3),(6,7),(10,11)]

    # Visualizer.plot_trees_together(T_i, T_f, "original_trees.png")
    # Visualizer.plot_linear_graph(T_i, T_f, T_i, T_f, "linear_graph.png")

    greed.get_flip_sequence(T_i, T_f)


if __name__ == "__main__":
    test_demo()
