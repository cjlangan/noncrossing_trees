import multiprocessing
from multiprocessing import Manager, Process, Event, Lock
from typing import Optional, Dict, Any

from .gamma import GammaAnalyzer
from ..core import TreeUtils
from ..generation import NCSTGenerator


class ParallelGammaSearcher:
    """Parallel searcher for gamma threshold optimization."""

    def __init__(self, num_workers: Optional[int] = None):
        self.num_workers = num_workers or multiprocessing.cpu_count()
        self.analyzer = GammaAnalyzer()

    def _worker_process(
        self, n: int, gamma_threshold: float, method: str,
        k: Optional[int], notable: bool, stop_event: Event,  # type: ignore
        result_holder: Dict, lock: Lock, worker_id: int  # type: ignore
    ):
        """Worker process for parallel gamma search."""
        local_tested = 0

        while not stop_event.is_set():
            try:
                # Generate first tree
                if k is None:
                    T_i, seed_i = NCSTGenerator.generate_random_ncst(n)
                else:
                    T_i, seed_i = NCSTGenerator.generate_ncst_with_k_borders(
                        n, k)

                local_tested += 1

                # Generate second tree based on method
                if method == "rf":
                    T_f = TreeUtils.flip_tree(
                        TreeUtils.rotate_tree(T_i, n // 2))
                    seed_f = f"{method}_{seed_i}"
                elif method == "fr":
                    T_f = TreeUtils.rotate_tree(
                        TreeUtils.flip_tree(T_i), n // 2)
                    seed_f = f"{method}_{seed_i}"
                elif method == "f":
                    T_f = TreeUtils.flip_tree(T_i)
                    seed_f = f"{method}_{seed_i}"
                elif method == "r":
                    T_f = TreeUtils.rotate_tree(T_i, n // 2)
                    seed_f = f"{method}_{seed_i}"
                else:  # random
                    if k is None:
                        T_f, seed_f = NCSTGenerator.generate_random_ncst(n)
                    else:
                        T_f, seed_f = NCSTGenerator.generate_ncst_with_k_borders(
                            n, k)

                # Analyze the tree pair
                curr_gamma, ac_h, E_i, E_f, H = self.analyzer.analyze_tree_pair(
                    T_i, T_f, verbose=False, plot=False)

                v_h = len(H.nodes) if H.nodes else 0

                # Check if we found a better gamma
                if curr_gamma is not None and curr_gamma <= gamma_threshold:
                    with lock:
                        # Double-check that no other worker found a better result
                        if ('curr_gamma' not in result_holder or
                                curr_gamma < result_holder['curr_gamma']):
                            result_holder['curr_gamma'] = curr_gamma
                            result_holder['ac_h'] = ac_h
                            result_holder['v_h'] = v_h
                            result_holder['T_i'] = T_i
                            result_holder['T_f'] = T_f
                            result_holder['seed_i'] = seed_i
                            result_holder['seed_f'] = seed_f
                            result_holder['E_i'] = E_i
                            result_holder['E_f'] = E_f
                            result_holder['H'] = H
                            result_holder['worker_id'] = worker_id

                        result_holder['total_tested'] += local_tested
                        local_tested = 0

                        if notable:
                            print(f"Worker {worker_id}: Found gamma of {curr_gamma:.6f} = "
                                  f"{ac_h}/{v_h} on {n} vertices with seeds {seed_i} and {seed_f}")

                        stop_event.set()  # Signal all workers to stop
                        break

                # Print notable results
                elif (notable and curr_gamma is not None and
                      (curr_gamma < 0.6 or local_tested % 10000 == 0)):
                    with lock:
                        print(f"Worker {worker_id}: Tested {local_tested + result_holder['total_tested']}, "
                              f"current gamma: {curr_gamma:.6f} = {ac_h}/{v_h}")

                # Periodically update total tested count
                if local_tested % 1000 == 0:
                    with lock:
                        result_holder['total_tested'] += local_tested
                        local_tested = 0

            except Exception as e:
                print(f"Worker {worker_id} encountered error: {e}")
                continue

        # Update final count
        with lock:
            result_holder['total_tested'] += local_tested

    def find_trees_with_gamma_parallel(self, n: int, gamma_threshold: float,
                                       method: str = "random", k: Optional[int] = None,
                                       notable: bool = True, plot: bool = True) -> Dict[str, Any]:
        """
        Parallel search for tree pairs meeting gamma threshold.

        Parameters:
            n: number of vertices
            gamma_threshold: Gamma threshold to find or better
            method: method to generate second NCST ('f', 'r', 'fr', 'rf', 'random')
            k: number of border edges (None for any)
            notable: whether to print notable gammas
            plot: whether to generate visualization plots

        Returns:
            Dictionary containing results or None if no suitable pair found
        """
        manager = Manager()
        stop_event = Event()
        result_holder = manager.dict()
        result_holder['total_tested'] = 0
        lock = Lock()

        print(f"Beginning parallel search for gamma ≤ {
              gamma_threshold} on {n} vertices...")
        print(
            f"Using {self.num_workers} worker processes with method '{method}'")
        if k is not None:
            print(f"Constraining to trees with exactly {k} border edges")

        # Start worker processes
        workers = []
        for worker_id in range(self.num_workers):
            p = Process(target=self._worker_process,
                        args=(n, gamma_threshold, method, k, notable,
                              stop_event, result_holder, lock, worker_id))
            p.start()
            workers.append(p)

        try:
            # Wait for all workers to complete
            for p in workers:
                p.join()
        except KeyboardInterrupt:
            print("\nSearch interrupted by user")
            stop_event.set()
            for p in workers:
                p.terminate()
                p.join()

        # Process results
        if 'curr_gamma' in result_holder:
            result = {
                'gamma': result_holder['curr_gamma'],
                'ac_h': result_holder['ac_h'],
                'v_h': result_holder['v_h'],
                'T_i': result_holder['T_i'],
                'T_f': result_holder['T_f'],
                'seed_i': result_holder['seed_i'],
                'seed_f': result_holder['seed_f'],
                'E_i': result_holder['E_i'],
                'E_f': result_holder['E_f'],
                'H': result_holder['H'],
                'total_tested': result_holder['total_tested'],
                'worker_id': result_holder['worker_id']
            }

            print(f"\n✅ SUCCESS: Found gamma of {result['gamma']:.6f} = "
                  f"{result['ac_h']}/{result['v_h']} on {n} vertices")
            print(f"Seeds: {result['seed_i']} and {result['seed_f']}")
            print(f"Total NCST pairs tested: {result['total_tested']}")
            print(f"Found by worker: {result['worker_id']}")

            if result['gamma'] < 4/9:
                print("🎉 EXCEPTIONAL RESULT: Found gamma < 4/9! 🎉")

            # Generate visualizations if requested
            if plot:
                print("\nGenerating visualization plots...")
                self.analyzer.visualizer.plot_trees_together(
                    result['T_i'], result['T_f'], "original_trees.png")
                self.analyzer.visualizer.plot_linear_graph(
                    result['T_i'], result['T_f'], result['E_i'], result['E_f'],
                    "linear_graph.png")
                self.analyzer.visualizer.plot_conflict_graph(
                    result['H'], "conflict_graph.png")

            return result
        else:
            print(f"\nNo gamma ≤ {gamma_threshold} found after testing {
                  result_holder['total_tested']} pairs")
            return {}
