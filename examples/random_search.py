from ..analysis import ParallelGammaSearcher
from ..visualization import Visualizer
from ..generation import NCSTGenerator

def random_search_demo():
    parallel_searcher = ParallelGammaSearcher()

    # parallel_searcher.find_trees_with_gamma_parallel(
    #         20,     # number of vertices
    #         .499,   # gamma value we want, or better
    #         "r",    # tree operation applied to second tree (flip)
    #         k = 4,  # number of border edges (optional)
    #         notable = True,     # option to print notable values
    #         plot = True         # option to plot graphs when done
    # )

    Visualizer.print_tree(NCSTGenerator.generate_ncst_with_k_borders(6, 2)[0])

if __name__ == "__main__":
    random_search_demo()
