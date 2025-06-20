from typing import List, Tuple, Set, FrozenSet, Any, Iterator
import multiprocessing
from multiprocessing.sharedctypes import Synchronized
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
import os
from ..analysis import GammaAnalyzer
from ..visualization import Visualizer
from ..formulas import NCSTFormulas
from ..generation import NecklaceGenerator, ConfinedEdgeGenerator
from ..core import TreeUtils
from ..optimization import OptimizationSolver

global_all_trees: List[List[Tuple[int, int]]] = []

class FiniteGammaSearcher:
    """Enumerative searcher for all NCSTs, with restrictions"""

    def __init__(self):
        self.gamma_analyzer = GammaAnalyzer()
        self.visualizer = Visualizer()
        self.optimizer = OptimizationSolver()
        self.shared_counter: Synchronized
        self.total_trees = None

    @staticmethod
    def rotated_versions(tree: list[tuple[int, int]]) -> list[frozenset[tuple[int, int]]]:
        """get all rotated versions of the initial tree"""
        n = len(tree) + 1
        return [
            frozenset(
                (min((a + r) % n, (b + r) % n), max((a + r) % n, (b + r) % n))
                for a, b in tree
            )
            for r in range(n)
        ]


    def enumerate_ncsts_k_borders(self, n: int, k: int, test: bool = True):
        """Enumerate over all NCSTs on n vertices with exactly k border edges"""
        seen: Set[FrozenSet[Tuple[int, int]]] = set()
        num_tested = 0
        best_gamma = 1
        best_trees: Tuple[List[Tuple[int, int]], List[Tuple[int, int]]] = ([], [])
        four_nine_gamma_counter = 0
        four_nine_list: List[List[Tuple[int, int]]] = []

        print(f"Beginning test on all NCSTSs with {n} vertices and {k} borders against their flip")
        num_trees = NCSTFormulas.T(n, k)
        Visualizer.print_progress_bar(num_tested, num_trees)

        def enumerate_ncsts_helper(
            points: List[int],
            local_edges: List[Tuple[int, int]],
            all_edges: Set[Tuple[int, int]],
        ):
            """Recursively chooses next chord for the left and right sections"""
            nonlocal num_tested, seen, best_gamma, best_trees, four_nine_gamma_counter

            if len(all_edges) == n - 1 and TreeUtils.is_valid_tree(list(range(n)), list(all_edges)):
                base_tree = [(min(a, b), max(a, b)) for a, b in all_edges]
                flipped_tree = TreeUtils.flip_tree(base_tree)

                for tree_variant in (base_tree, flipped_tree):
                    for rotated in self.rotated_versions(tree_variant):
                        if rotated in seen:
                            continue
                        seen.add(rotated)
                        num_tested += 1

                        to_test: List[Tuple[int, int]] = list(rotated)

                        if test:
                            gamma, ac_h, E_i, E_f, H = self.gamma_analyzer.analyze_tree_pair(
                                to_test, TreeUtils.flip_tree(to_test), verbose=False, plot=False
                            )

                            self.visualizer.print_progress_bar(num_tested, num_trees)

                            if gamma is None:
                                continue 

                            if gamma < best_gamma:
                                best_gamma = gamma
                                best_trees = (to_test, TreeUtils.flip_tree(to_test))

                            if gamma <= 4 / 9:
                                four_nine_gamma_counter += 1
                                four_nine_list.append(to_test)

                            if gamma < 4 / 9:
                                print("\n✅✅✅FOUND BETTER GAMMA✅✅✅", flush=True)
                                print(f"Gamma = {gamma}", flush=True)
                                print(f"Used {sorted(to_test)} and flip", flush=True)
                                exit()
                return

            for i in range(len(points)):
                for j in range(i + 2, len(points)):
                    a, b = points[i], points[j]
                    if (a + 1) % n == b or (b + 1) % n == a:
                        continue

                    edge = (min(a, b), max(a, b))
                    if edge in all_edges:
                        continue

                    new_local_edges = local_edges + [edge]
                    new_all_edges = all_edges | {edge}

                    if TreeUtils.has_cycle(new_local_edges):
                        continue

                    a_idx, b_idx = sorted((points.index(a), points.index(b)))
                    between = points[a_idx + 1:b_idx]
                    outside = points[:a_idx + 1] + points[b_idx:]

                    between_edges = [e for e in new_local_edges if e[0] in between and e[1] in between]
                    outside_edges = [e for e in new_local_edges if e[0] in outside and e[1] in outside]

                    enumerate_ncsts_helper(between, between_edges, new_all_edges)
                    enumerate_ncsts_helper(outside, outside_edges, new_all_edges)

        for necklace in NecklaceGenerator.generate_binary_necklaces(n, k, reflective=False):
            borders = [(i, (i + 1) % n) for i, b in enumerate(necklace) if b == 1]
            border_set: Set[Tuple[int, int]] = {(min(a, b), max(a, b)) for a, b in borders}
            enumerate_ncsts_helper(list(range(n)), borders, border_set)

        print(f"\nTesting complete. Total NCSTSs tested: {num_tested}")
        print(f"Best gamma: {best_gamma}")
        print(f"Found on trees: {best_trees[0]} and {best_trees[1]}")
        print(f"Number of 4/9 or better: {four_nine_gamma_counter}")
        print(f"All 4/9 or better: {four_nine_list}")


    def init_worker(self, counter: Synchronized, total: int) -> None:
        """Initialize worker process with shared variables"""
        global shared_counter, total_trees
        shared_counter = counter
        total_trees = total


    def chunkify(self, iterable: List[Any], n: int) -> Iterator[List[Any]]:
        """Split iterable into n chunks as evenly as possible."""
        items = list(iterable)
        k, m = divmod(len(items), n)
        return (items[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


    def worker_process(self, n: int, k: int, necklace_batch: List[List[int]], test: bool) -> Tuple[int, float, Tuple[List[Tuple[int, int]], List[Tuple[int, int]]], int, List[FrozenSet[Tuple[int, int]]]]:
        """Worker process for parallel NCST enumeration"""
        global shared_counter, total_trees
        seen: Set[FrozenSet[Tuple[int, int]]] = set()
        local_best_gamma = 1.0
        local_best_trees: Tuple[List[Tuple[int, int]], List[Tuple[int, int]]] = ([], [])
        local_four_nine_gamma_counter = 0
        local_four_nine_list: List[FrozenSet[Tuple[int, int]]] = []
        local_num_tested = 0

        # Initialize analyzer for this worker
        gamma_analyzer = GammaAnalyzer()

        def enumerate_ncsts_helper(
            points: List[int], 
            local_edges: List[Tuple[int, int]], 
            all_edges: Set[Tuple[int, int]]
        ) -> None:
            nonlocal local_num_tested, seen, local_best_gamma, local_best_trees, local_four_nine_gamma_counter

            if len(all_edges) == n - 1 and TreeUtils.is_valid_tree(list(range(n)), list(all_edges)):
                base_tree = [(min(a, b), max(a, b)) for a, b in all_edges]
                flipped_tree = TreeUtils.flip_tree(base_tree)

                for tree_variant in (base_tree, flipped_tree):
                    for rotated in self.rotated_versions(tree_variant):
                        if rotated in seen:
                            continue
                        seen.add(rotated)
                        local_num_tested += 1

                        # Update shared counter and print progress
                        with shared_counter.get_lock():
                            shared_counter.value += 1
                            if shared_counter.value % 10 == 0:
                                Visualizer.print_progress_bar(shared_counter.value, total_trees)

                        gamma = None
                        if test:
                            gamma, *_ = gamma_analyzer.analyze_tree_pair(
                                list(rotated), TreeUtils.flip_tree(list(rotated)), verbose=False, plot=False
                            )

                        if test and gamma is not None:
                            if gamma < local_best_gamma:
                                local_best_gamma = gamma
                                local_best_trees = (sorted(rotated), sorted(TreeUtils.flip_tree(list(rotated))))
                            if gamma <= 4 / 9:
                                local_four_nine_gamma_counter += 1
                                local_four_nine_list.append(rotated)
                return

            for i in range(len(points)):
                for j in range(i + 2, len(points)):
                    a, b = points[i], points[j]
                    if (a + 1) % n == b or (b + 1) % n == a:
                        continue
                    edge = (min(a, b), max(a, b))
                    if edge in all_edges:
                        continue
                    new_edge = (a, b)
                    new_local_edges = local_edges + [new_edge]
                    new_all_edges = all_edges | {edge}
                    if TreeUtils.has_cycle(new_local_edges):
                        continue
                    a_idx, b_idx = sorted((points.index(a), points.index(b)))
                    between = points[a_idx + 1:b_idx]
                    outside = points[:a_idx + 1] + points[b_idx:]
                    between_edges = [e for e in new_local_edges if e[0] in between and e[1] in between]
                    outside_edges = [e for e in new_local_edges if e[0] in outside and e[1] in outside]
                    enumerate_ncsts_helper(between, between_edges, new_all_edges)
                    enumerate_ncsts_helper(outside, outside_edges, new_all_edges)

        for necklace in necklace_batch:
            borders = [(i, (i + 1) % n) for i, b in enumerate(necklace) if b == 1]
            border_set: Set[Tuple[int, int]] = {(min(a, b), max(a, b)) for a, b in borders}
            enumerate_ncsts_helper(list(range(n)), borders, border_set)

        return local_num_tested, local_best_gamma, local_best_trees, local_four_nine_gamma_counter, local_four_nine_list


    def worker_wrapper(self, args: Tuple[int, int, List[List[int]], bool]) -> Tuple[int, float, Tuple[List[Tuple[int, int]], List[Tuple[int, int]]], int, List[FrozenSet[Tuple[int, int]]]]:
        """Define a separate function that can be pickled"""
        n, k, necklace_batch, test = args
        return self.worker_process(n, k, necklace_batch, test)


    def enumerate_ncsts_k_borders_parallel(self, n: int, k: int, test: bool = True) -> None:
        """
        Function to enumerate over all NCSTs with exactly k borders using parallel processing.
        
        Parameters:
            n: number of vertices
            k: number of borders
            test: if False, then just enumerates without testing for gammas
        
        Note: Currently this function only tests a found tree against its flip counterpart.
              That is, it does NOT test every conceivable combination.
        """
        print(f"Parallel NCST Search for n={n}, k={k}, testing against flip.")
        total_trees_expected = NCSTFormulas.T(n, k)
        print(f"Expected total trees: {total_trees_expected}")
        
        all_necklaces = list(NecklaceGenerator.generate_binary_necklaces(n, k, reflective=True))

        cpu_count = multiprocessing.cpu_count()
        necklace_chunks = list(self.chunkify(all_necklaces, cpu_count))
        
        # Create shared counter for progress tracking
        counter = multiprocessing.Value('i', 0)

        # Create argument tuples for the worker function
        worker_args = [(n, k, chunk, test) for chunk in necklace_chunks]
        
        Visualizer.print_progress_bar(0, total_trees_expected)

        with multiprocessing.Pool(cpu_count, initializer=self.init_worker, initargs=(counter, total_trees_expected)) as pool:
            results = list(pool.map(self.worker_wrapper, worker_args))
            
        # Final progress update
        with counter.get_lock():
            Visualizer.print_progress_bar(counter.value, counter.value)  # Show 100%
        print()  # New line after progress bar

        total_tested = 0
        best_gamma = 1.0
        best_trees: Tuple[List[Tuple[int, int]], List[Tuple[int, int]]] = ([], [])
        total_four_nine = 0
        all_four_nine: List[FrozenSet[Tuple[int, int]]] = []

        for tested, gamma, trees, count_49, list_49 in results:
            total_tested += tested
            if gamma < best_gamma:
                best_gamma = gamma
                best_trees = trees
            total_four_nine += count_49
            all_four_nine.extend(list_49)

        print(f"\nTesting complete. Total NCSTSs tested: {total_tested}")
        print(f"Best gamma: {best_gamma}")
        print(f"Found on trees: {best_trees[0]} and {best_trees[1]}")
        print(f"Number of 4/9 or better: {total_four_nine}")
        print(f"All 4/9 or better: {all_four_nine}")


    @staticmethod
    def even_rotated_versions(tree: list[tuple[int, int]], k: int) -> list[frozenset[tuple[int, int]]]:
        """Get all rotated versions of the tree such that k evenly spaced borders remain in place."""
        n = len(tree) + 1
        step = n // k
        return [
            frozenset(
                (min((a + r) % n, (b + r) % n), max((a + r) % n, (b + r) % n))
                for a, b in tree
            )
            for r in range(0, n, step)
        ]

    def enumerate_even_shared_full(self, n: int, k: int, test: bool = True, plot: bool = True):
        """
        Generate all NCSTs with k evenly spaced border edges and compare every unique pair
        that preserves the shared border edges.
        """
        print(f"Generating all NCSTs with {n} vertices and {k} evenly-spaced shared border edges...")

        seen: Set[frozenset[Tuple[int, int]]] = set()
        all_trees: List[List[Tuple[int, int]]] = []

        def enumerate_ncsts_helper(
            points: List[int],
            local_edges: List[Tuple[int, int]],
            all_edges: Set[Tuple[int, int]],
        ):
            """Recursively chooses next chord for the left and right sections"""
            if len(all_edges) == n - 1 and TreeUtils.is_valid_tree(list(range(n)), list(all_edges)):
                base_tree = [(min(a, b), max(a, b)) for a, b in all_edges]

                for variant in (base_tree, TreeUtils.flip_tree(base_tree)):
                    for rotated in self.even_rotated_versions(variant, k):
                        if rotated in seen:
                            continue
                        seen.add(rotated)
                        all_trees.append(sorted(list(rotated)))
                return

            for i in range(len(points)):
                for j in range(i + 2, len(points)):
                    a, b = points[i], points[j]
                    if (a + 1) % n == b or (b + 1) % n == a:
                        continue

                    edge = (min(a, b), max(a, b))
                    if edge in all_edges:
                        continue

                    new_local_edges = local_edges + [edge]
                    new_all_edges = all_edges | {edge}

                    if TreeUtils.has_cycle(new_local_edges):
                        continue

                    a_idx, b_idx = sorted((points.index(a), points.index(b)))
                    between = points[a_idx + 1:b_idx]
                    outside = points[:a_idx + 1] + points[b_idx:]

                    between_edges = [e for e in new_local_edges if e[0] in between and e[1] in between]
                    outside_edges = [e for e in new_local_edges if e[0] in outside and e[1] in outside]

                    enumerate_ncsts_helper(between, between_edges, new_all_edges)
                    enumerate_ncsts_helper(outside, outside_edges, new_all_edges)

        # Generate all valid trees
        borders = ConfinedEdgeGenerator.evenly_spaced_border_combination(n, k)
        border_set = {(min(a, b), max(a, b)) for a, b in borders}
        enumerate_ncsts_helper(list(range(n)), borders, set(border_set))

        print(f"\nGenerated {len(all_trees)} unique NCSTs with shared borders.")
        print(f"Beginning pairwise comparisons...")

        # Analysis Phase
        best_gamma = 1
        best_trees: Tuple[List[Tuple[int, int]], List[Tuple[int, int]]] = ([], [])
        four_nine_gamma_counter = 0
        four_nine_list: List[List[Tuple[int, int]]] = []

        num_pairs = 0
        pair_counter = 0
        for i in range(len(all_trees)):
            for j in range(i + 1, len(all_trees)):
                if TreeUtils.trees_share_borders(all_trees[i], all_trees[j]):
                    num_pairs += 1

        Visualizer.print_progress_bar(0, num_pairs)

        for i in range(len(all_trees)):
            for j in range(i + 1, len(all_trees)):
                if not TreeUtils.trees_share_borders(all_trees[i], all_trees[j]):
                    continue
                t1 = all_trees[i]
                t2 = all_trees[j]
                pair_counter += 1

                if test:
                    gamma, ac_h, E_i, E_f, H = self.gamma_analyzer.analyze_tree_pair(
                        t1, t2, verbose=False, plot=False
                    )

                    self.visualizer.print_progress_bar(pair_counter, num_pairs)

                    if gamma is None:
                        continue

                    if gamma < best_gamma:
                        best_gamma = gamma
                        best_trees = (t1, t2)

                    if gamma <= 4 / 9:
                        four_nine_gamma_counter += 1
                        four_nine_list.append(t1)

                    if gamma < 4 / 9:
                        print("\n✅✅✅FOUND BETTER GAMMA✅✅✅", flush=True)
                        print(f"Gamma = {gamma}", flush=True)
                        print(f"Used {sorted(t1)} and {sorted(t2)}", flush=True)
                        exit()

        print(f"\nPairwise comparison complete.")
        print(f"Total pairs tested: {pair_counter}")
        print(f"Best gamma: {best_gamma}")
        print(f"Found on trees: {best_trees[0]} and {best_trees[1]}")
        print(f"Number of 4/9 or better: {four_nine_gamma_counter}")
        print(f"All 4/9 or better: {four_nine_list}")

        if test and plot:
            self.gamma_analyzer.analyze_tree_pair(best_trees[0], best_trees[1], verbose=False)


    def enumerate_even_shared_full_parallel(self, n: int, k: int, test: bool = True, plot: bool = True):
        print(f"Generating all NCSTs with {n} vertices and {k} evenly-spaced shared border edges...")

        seen: Set[frozenset[Tuple[int, int]]] = set()
        all_trees: List[List[Tuple[int, int]]] = []

        def enumerate_ncsts_helper(points, local_edges, all_edges):
            if len(all_edges) == n - 1 and TreeUtils.is_valid_tree(list(range(n)), list(all_edges)):
                base_tree = [(min(a, b), max(a, b)) for a, b in all_edges]
                for variant in (base_tree, TreeUtils.flip_tree(base_tree)):
                    for rotated in self.even_rotated_versions(variant, k):
                        if rotated in seen:
                            continue
                        seen.add(rotated)
                        all_trees.append(sorted(list(rotated)))
                return

            for i in range(len(points)):
                for j in range(i + 2, len(points)):
                    a, b = points[i], points[j]
                    if (a + 1) % n == b or (b + 1) % n == a:
                        continue
                    edge = (min(a, b), max(a, b))
                    if edge in all_edges:
                        continue
                    new_local_edges = local_edges + [edge]
                    new_all_edges = all_edges | {edge}
                    if TreeUtils.has_cycle(new_local_edges):
                        continue
                    a_idx, b_idx = sorted((points.index(a), points.index(b)))
                    between = points[a_idx + 1:b_idx]
                    outside = points[:a_idx + 1] + points[b_idx:]
                    between_edges = [e for e in new_local_edges if e[0] in between and e[1] in between]
                    outside_edges = [e for e in new_local_edges if e[0] in outside and e[1] in outside]
                    enumerate_ncsts_helper(between, between_edges, new_all_edges)
                    enumerate_ncsts_helper(outside, outside_edges, new_all_edges)

        borders = ConfinedEdgeGenerator.evenly_spaced_border_combination(n, k)
        border_set = {(min(a, b), max(a, b)) for a, b in borders}
        enumerate_ncsts_helper(list(range(n)), borders, set(border_set))

        print(f"\nGenerated {len(all_trees)} unique NCSTs with shared borders.")
        print("Beginning pairwise comparisons...")

        # Prepare all pairs once
        
        dict = {}
        for i in range(len(all_trees)):
            border_edges = frozenset(TreeUtils.get_border_edges(all_trees[i], n))
            if border_edges not in dict:
                dict[border_edges] = []
            dict[border_edges].append(i)
        pairs = []
        for border_edges, indices in dict.items():
            if len(indices) < 2:
                continue
            for i, j in combinations(indices, 2):
                if TreeUtils.trees_share_borders(all_trees[i], all_trees[j]):
                    pairs.append((i, j))
        num_pairs = len(pairs)
        print(f"Total pairs to compare: {num_pairs}")

        # Shared state
        best_result = {
            "gamma": 1,
            "pair": ([], []),
            "four_nine_count": 0,
            "four_nine_list": []
        }

        tasks = pairs

        print(f"Starting pairwise comparisons in parallel...")

        with ProcessPoolExecutor(initializer=init_workers, initargs=(all_trees,)) as executor:
            for idx, result in enumerate(executor.map(analyze_tree_pair_static, tasks)):
                i, j, gamma = result
                Visualizer.print_progress_bar(idx + 1, len(tasks))

                if gamma is None:
                    continue

                t1, t2 = all_trees[i], all_trees[j]

                if gamma < best_result["gamma"]:
                    best_result["gamma"] = gamma
                    best_result["pair"] = (t1, t2)

                if gamma <= 4 / 9:
                    best_result["four_nine_count"] += 1
                    best_result["four_nine_list"].append(t1)

                if gamma < 4 / 9:
                    print("\n✅✅✅FOUND BETTER GAMMA✅✅✅", flush=True)
                    print(f"Gamma = {gamma}", flush=True)
                    print(f"Used {sorted(t1)} and {sorted(t2)}", flush=True)
                    exit()

        print(f"\nPairwise comparison complete.")
        print(f"Total pairs tested: {num_pairs}")
        print(f"Best gamma: {best_result['gamma']}")
        print(f"Found on trees: {best_result['pair'][0]} and {best_result['pair'][1]}")
        print(f"Number of 4/9 or better: {best_result['four_nine_count']}")
        print(f"All 4/9 or better: {best_result['four_nine_list']}")

        if test and plot:
            self.gamma_analyzer.analyze_tree_pair(
                best_result["pair"][0], best_result["pair"][1], verbose=False
            )

# Init workers with global_all_trees
def init_workers(trees: List[List[Tuple[int, int]]]):
    global global_all_trees
    global_all_trees = trees

def analyze_tree_pair_static(args):
    i, j = args
    t1, t2 = global_all_trees[i], global_all_trees[j]
    gamma_analyzer = GammaAnalyzer()  # If you can safely instantiate it here
    gamma, _, _, _, _ = gamma_analyzer.analyze_tree_pair(t1, t2, verbose=False, plot=False)
    return (i, j, gamma)
