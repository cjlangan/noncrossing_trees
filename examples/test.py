from ..analysis import Greedy, ConflictAnalyzer
from ..generation import NCSTGenerator
from ..visualization import Visualizer
from ..core import TreeUtils

def test_demo():
    greed = Greedy()
    gen = NCSTGenerator()
    ana = ConflictAnalyzer()

    # T_i = gen.generate_random_ncst(20)[0]
    # T_f = gen.generate_random_ncst(20)[0]

    # Paper trees
    # T_i = [(0,3),(0,4),(1,3),(4,7),(4,8),(5,7),(8,11),(8,12),(9,11),(2, 3),(6,7),(10,11)]
    # T_f = [(0,10),(1,10),(1,5),(2,4),(2,5),(5,9),(6,8),(6,9),(10,12),(2,3),(6,7),(10,11)]

    # Symmetric n=13 pair
    T_i = [(0,3),(3,12),(1,3),(4,7),(4,8),(5,7),(8,11),(8,12),(9,11),(2, 3),(6,7),(10,11)]
    T_f = [(0,10),(1,10),(1,5),(2,4),(2,5),(5,9),(6,8),(6,9),(10,12),(2,3),(6,7),(10,11)]

    # This is the best pair of trees (lowest known gamma with the lowest number of vertices)
    # T_i = [(5,9),(0,9),(1,4),(0,11),(1,5),(7,8),(2,4),(3,4),(6,8),(5,8),(0,10)]
    # T_f = [(6,2),(11,2),(10,7),(11,0),(10,6),(4,3),(9,7),(8,7),(5,3),(6,3),(11,1)]

    conflict_vertices, E_i, E_f = ana.get_gaps_and_edge_pairs(T_i, T_f)

    # Get the k blowups
    T_i_blown = TreeUtils.blowup_tree(T_i, 0, E_i, conflict_vertices)
    T_f_blown = TreeUtils.blowup_tree(T_f, 0, E_f, conflict_vertices)

    # Visualizer.plot_trees_together(T_i, T_f, "original_trees.png")
    # Visualizer.plot_linear_graph(T_i, T_f, T_i, T_f, "linear_graph.png")

    for i in range(len(T_i_blown) + 1):
        sequence = greed.get_flip_sequence(
                TreeUtils.rotate_tree(T_i_blown, i), 
                TreeUtils.rotate_tree(T_f_blown, i))
        print("Flip distance:", len(sequence))


    T_ir = TreeUtils.rotate_tree(T_i_blown, 0)
    T_fr = TreeUtils.rotate_tree(T_f_blown, 0)
    Visualizer.plot_trees_together(T_ir, T_fr, "original_trees.png")
    sequence = greed.get_flip_sequence(T_ir, T_fr, verbose=True, slow=True)
    print("Flip distance:", len(sequence))

if __name__ == "__main__":
    test_demo()
