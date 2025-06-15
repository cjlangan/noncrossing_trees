from ..analysis import GammaAnalyzer, ParallelGammaSearcher


def run_demo():
    # Initialize the main analyzer
    gamma_analyzer = GammaAnalyzer()
    parallel_searcher = ParallelGammaSearcher(num_workers=6)

    # Trees from the paper, n = 13
    # T_i = [(0,3),(0,4),(1,3),(4,7),(4,8),(5,7),(8,11),(8,12),(9,11),(2, 3),(6,7),(10,11)]
    # T_f = [(0,10),(1,10),(1,5),(2,4),(2,5),(5,9),(6,8),(6,9),(10,12),(2,3),(6,7),(10,11)]

    # Paper Trees twice extended
    # T_f = [(0,18),(1,18),(1,5),(2,4),(2,5),(5,9),(6,8),(6,9),(10,12),(2,3),(6,7),(10,11),(9,13),(10,13),(14,15),(14,16),(13,17),(14,17),(18,19),(18,20)]
    # T_i = [(0,3),(0,4),(1,3),(4,7),(4,8),(5,7),(8,11),(8,12),(9,11),(2, 3),(6,7),(10,11),(12,15),(12,16),(13,15),(14,15),(16,19),(16,20),(17,19),(18,19)]

    # Paper Trees +2 edge
    # T_i = [(0,3),(0,4),(1,3),(4,7),(4,8),(5,7),(8,11),(8,12),(9,11),(2, 3),(6,7),(10,11),(8,13),(8,14)]
    # T_f = [(0,10),(1,10),(1,5),(2,4),(2,5),(5,9),(6,8),(6,9),(10,12),(2,3),(6,7),(10,11),(12,13),(13,14)]

    # Trees I found with gamma = .45454545
    # T_i, _ = NCSTGenerator.generate_random_ncst(17, seed=355920981)
    # T_f = TreeUtils.flip_tree(T_i)

    # Trees I found with gamma = .44444444
    # T_i = [(8, 14), (3, 15), (9, 11), (3, 8), (9, 13), (0, 15), (5, 8), (5, 7), (4, 8), (2, 15), (9, 12), (1, 15), (10, 11), (13, 14), (5, 6)]
    # T_f = [(7, 1), (12, 0), (6, 4), (12, 7), (6, 2), (15, 0), (10, 7), (10, 8), (11, 7), (13, 0), (6, 3), (14, 0), (5, 4), (2, 1), (10, 9)]

    # Trees I found with gamma = .44444444, n = 12
    # This is the best pair of trees (lowest known gamma with the lowest number of vertices)
    T_i = [(5, 9), (0, 9), (1, 4), (0, 11), (1, 5), (7, 8),
           (2, 4), (3, 4), (6, 8), (5, 8), (0, 10)]
    T_f = [(6, 2), (11, 2), (10, 7), (11, 0), (10, 6),
           (4, 3), (9, 7), (8, 7), (5, 3), (6, 3), (11, 1)]

    # Best tree with added cell, no good
    # T_i = [(5, 9), (0, 13), (1, 4), (0, 15), (1, 5), (7, 8), (2, 4), (3, 4), (6, 8), (5, 8), (0, 14), (9, 13), (9, 12), (10, 12), (11, 12)]
    # T_f = TreeUtils.flip_tree(T_i)

    # Nam Trees with edge removal
    # T_i = [(3, 4), (10, 11), (5, 6), (2, 15), (4, 6), (8, 14), (0, 15), (1, 15), (9, 14), (3, 7), (2, 8), (3, 8), (9, 12), (10, 12), (9, 13)]
    # T_f = [(12, 11), (5, 4), (10, 9), (13, 0), (11, 9), (7, 1), (15, 0), (14, 0), (6, 1), (12, 8), (13, 7), (12, 7), (6, 3), (5, 3), (6, 2)]

    # Analyze the best known tree pair
    print("=== Analyzing Best Known Tree Pair (gamma ≈ 0.44444444, n=12) ===")
    result = gamma_analyzer.analyze_tree_pair(
        T_i, T_f, verbose=True, plot=True)
    gamma, ac_h, E_i, E_f, H = result
    print(f"Analysis complete: gamma = {gamma}")

    # Alternative analyses you can uncomment:

    # Analyze rotated versions
    # print("\n=== Analyzing Rotated Tree Pair ===")
    # T_i_rotated = TreeUtils.rotate_tree(T_i, 1)
    # T_f_rotated = TreeUtils.rotate_tree(T_f, 1)
    # gamma_analyzer.analyze_tree_pair(T_i_rotated, T_f_rotated, verbose=True, plot=False)

    # Search for better trees using parallel search
    # print("\n=== Searching for Trees with gamma < 0.5 ===")
    # result = parallel_searcher.find_trees_with_gamma_parallel(
    #     n=12,
    #     gamma_threshold=0.5,
    #     method="f",  # flip method
    #     k=3,         # exactly 3 border edges
    #     notable=True,
    #     plot=True
    # )

    # Generate and analyze a random NCST pair
    # print("\n=== Generating and Analyzing Random NCST Pair ===")
    # T_random1, seed1 = NCSTGenerator.generate_random_ncst(12)
    # T_random2, seed2 = NCSTGenerator.generate_random_ncst(12)
    # print(f"Generated trees with seeds: {seed1}, {seed2}")
    # gamma_analyzer.analyze_tree_pair(T_random1, T_random2, verbose=True, plot=False)

    # Generate NCST with specific number of border edges
    # print("\n=== Generating NCST with 3 Border Edges ===")
    # try:
    #     T_border, seed = NCSTGenerator.generate_ncst_with_k_borders(12, 3)
    #     T_border_flipped = TreeUtils.flip_tree(T_border)
    #     print(f"Generated tree with 3 borders, seed: {seed}")
    #     gamma_analyzer.analyze_tree_pair(T_border, T_border_flipped, verbose=True, plot=False)
    # except RuntimeError as e:
    #     print(f"Could not generate tree with 3 borders: {e}")

    # Search using different methods
    # methods_to_try = ["f", "r", "rf", "fr", "random"]
    # for method in methods_to_try:
    #     print(f"\n=== Searching with method '{method}' ===")
    #     result = gamma_analyzer.search_for_gamma_threshold(
    #         n=10,
    #         gamma_threshold=0.6,
    #         method=method,
    #         k=None,  # any number of border edges
    #         verbose=False,
    #         plot=False
    #     )
    #     if result['gamma'] is not None:
    #         print(f"Method '{method}': Found gamma = {result['gamma']:.6f} after {result['num_tested']} tests")

    # Note: The enumerate_ncsts_k_borders_parallel function mentioned in your original code
    # is not implemented in the provided classes. You would need to implement this separately
    # or use the existing parallel search functionality shown above.

    print("\n=== Analysis Complete ===")
    print("Check the generated PNG files for visualizations:")
    print("- original_trees.png: Shows both trees on convex position")
    print("- linear_graph.png: Shows linear representation")
    print("- conflict_graph.png: Shows conflict graph with edge types")


if __name__ == "__main__":
    run_demo()
