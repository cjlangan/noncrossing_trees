from ..analysis import ParallelGammaSearcher

def random_search_demo():
    parallel_searcher = ParallelGammaSearcher()

    parallel_searcher.find_trees_with_gamma_parallel(
            13,     # number of vertices
            .58,   # gamma value we want, or better
            "f",    # tree operation applied to second tree (flip)
            k = 3,  # number of border edges (optional)
            notable = True,     # option to print notable values
            plot = True         # option to plot graphs when done
    )

if __name__ == "__main__":
    random_search_demo()
