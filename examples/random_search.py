from ..analysis import ParallelGammaSearcher
from ..generation import ConfinedEdgeGenerator

def random_search_demo():
    parallel_searcher = ParallelGammaSearcher()

    bord = ConfinedEdgeGenerator.evenly_spaced_border_combination(12, 3)

    parallel_searcher.find_trees_with_gamma_parallel(
            12,     # number of vertices
            .6,   # gamma value we want, or better
            method="f",         # tree operation applied to second tree (flip)
            # k = 3,            # number of border edges (optional)
            borders = bord,     # optionally get specific border edges
            # borders = [(2, 3), (6, 7), (10, 11)],
            notable = True,     # option to print notable values
            skip_half = True,   # option skip print .5 gammas
            plot = True         # option to plot graphs when done
    )

if __name__ == "__main__":
    random_search_demo()
