from ..analysis import Greedy
from ..visualization import Visualizer
from ..generation import NCSTGenerator

def greedy_random_demo():
    greed = Greedy()
    gen = NCSTGenerator()

    print("==== BEGINNING DEMO ====\n")

    print("Number of vertices: ")
    n = int(input())
    print("Number of borders each: ")
    k = int(input())

    # Get the random trees
    T_i, seed_i = gen.generate_ncst_with_k_borders(n, k)
    T_f, seed_f = gen.generate_ncst_with_k_borders(n, k)
    sequence = greed.get_flip_sequence(T_i, T_f, slow=False, verbose=False)
    print(f"Seeds: {seed_i}, {seed_f}")
    print(f"Flip distance: {len(sequence)}")

    # Plot the initial trees
    Visualizer.plot_trees_together(T_i, T_f, "original_trees.png")
    Visualizer.plot_linear_graph(T_i, T_f, T_i, T_f, "linear_graph.png")

    # Go throgh the chosen pair slowly
    sequence = greed.get_flip_sequence(T_i, T_f, verbose=True, slow=True)

    print(f"Final sequence: ")
    print(sequence)

    print("\n==== DEMO COMPLETE ====")

if __name__ == "__main__":
    greedy_random_demo()

