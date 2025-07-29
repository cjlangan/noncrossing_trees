from ..analysis import Greedy, ConflictAnalyzer
from ..visualization import Visualizer
from ..core import TreeUtils

def greedy_best_trees_demo():
    greed = Greedy()
    ana = ConflictAnalyzer()

    print("==== BEGINNING DEMO ====\n")

    print("""Which best pair do you want to test?
        0: Trees from Bjerkevik's Paper
        1: Symmetric n=13 pair
        2: n=12 pair""")
    choice = int(input())

    if choice == 0:
        # Trees from the paper, n = 13
        T_i = [(0,3),(0,4),(1,3),(4,7),(4,8),(5,7),(8,11),(8,12),(9,11),(2, 3),(6,7),(10,11)]
        T_f = [(0,10),(1,10),(1,5),(2,4),(2,5),(5,9),(6,8),(6,9),(10,12),(2,3),(6,7),(10,11)]
    elif choice == 1:
        # New symmetric pair, n = 13
        T_i = [(0,3),(3,12),(1,3),(4,7),(4,8),(5,7),(8,11),(8,12),(9,11),(2, 3),(6,7),(10,11)]
        T_f = [(0,10),(1,10),(1,5),(2,4),(2,5),(5,9),(6,8),(6,9),(10,12),(2,3),(6,7),(10,11)]
    elif choice == 2:
        # This is the best pair of trees (lowest known gamma with the lowest number of vertices)
        T_i = [(5,9),(0,9),(1,4),(0,11),(1,5),(7,8),(2,4),(3,4),(6,8),(5,8),(0,10)]
        T_f = [(6,2),(11,2),(10,7),(11,0),(10,6),(4,3),(9,7),(8,7),(5,3),(6,3),(11,1)]
    else:
        # Default to symmetric n=13 pair
        T_i = [(0,3),(3,12),(1,3),(4,7),(4,8),(5,7),(8,11),(8,12),(9,11),(2, 3),(6,7),(10,11)]
        T_f = [(0,10),(1,10),(1,5),(2,4),(2,5),(5,9),(6,8),(6,9),(10,12),(2,3),(6,7),(10,11)]

    print("What k-blowup do you want to test? (0 for none)")
    k = int(input())
    assert(0 <= k <= 100)

    # Get the k blowups
    conflict_vertices, E_i, E_f = ana.get_gaps_and_edge_pairs(T_i, T_f)
    T_i_blown = TreeUtils.blowup_tree(T_i, k, E_i, conflict_vertices)
    T_f_blown = TreeUtils.blowup_tree(T_f, k, E_f, conflict_vertices)

    # Plot the initial trees
    Visualizer.plot_trees_together(T_i, T_f, "original_trees.png")
    Visualizer.plot_linear_graph(T_i, T_f, T_i, T_f, "linear_graph.png")

    # Calculate flip distance for every rotation
    for i in range(len(T_i_blown) + 1):
        sequence = greed.get_flip_sequence(
                TreeUtils.rotate_tree(T_i_blown, i), 
                TreeUtils.rotate_tree(T_f_blown, i))
        print(f"Rotation {i}: Flip distance = {len(sequence)}")

    print("Which rotation would you like to go through slowly? (0 for none)")
    rotation = int(input())
    assert(0 <= rotation <= len(T_i_blown))

    # Go throgh the chosen pair slowly
    T_ir = TreeUtils.rotate_tree(T_i_blown, rotation)
    T_fr = TreeUtils.rotate_tree(T_f_blown, rotation)
    Visualizer.plot_trees_together(T_ir, T_fr, "original_trees.png")
    sequence = greed.get_flip_sequence(T_ir, T_fr, verbose=False, slow=False)
    print("Flip distance:", len(sequence))
    sequence = greed.get_flip_sequence(T_ir, T_fr, verbose=True, slow=True)

    print(f"Final sequence: ")
    print(sequence)

    print("\n==== DEMO COMPLETE ====")

if __name__ == "__main__":
    greedy_best_trees_demo()

