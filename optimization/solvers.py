import networkx as nx
from typing import List
from ortools.sat.python import cp_model
import math

class OptimizationSolver:
    """Solver for optimization problems on conflict graphs."""

    @staticmethod
    def find_largest_acyclic_subgraph(H: nx.DiGraph, gamma: float = 4/9) -> List[int]:
        """Find the largest acyclic subgraph using CP-SAT solver from OR-Tools.
        Stops early if an acyclic subgraph with size > gamma * n is found. 
        Set gamma to 1.0 to disable threshold."""

        class ThresholdStopCallback(cp_model.CpSolverSolutionCallback):
            def __init__(self, x_vars, threshold):
                super().__init__()
                self.x_vars = x_vars
                self.threshold = threshold
                self.best_solution = None

            def on_solution_callback(self):
                current_size = sum(self.Value(x) for x in self.x_vars.values())
                if current_size > self.threshold:
                    self.best_solution = [v for v, x in self.x_vars.items() if self.Value(x) > 0]
                    self.StopSearch()

        model = cp_model.CpModel()
        n = len(H.nodes)
        nodes = list(H.nodes)
        threshold = math.ceil(gamma * n)

        x = {v: model.NewBoolVar(f"x_{v}") for v in nodes}
        order = {v: model.NewIntVar(0, n - 1, f"order_{v}") for v in nodes}

        model.Maximize(sum(x[v] for v in nodes))

        M = n
        for u, v in H.edges():
            model.Add(order[u] + 1 <= order[v] + M * (2 - x[u] - x[v]))

        solver = cp_model.CpSolver()
        callback = ThresholdStopCallback(x, threshold)
        status = solver.SolveWithSolutionCallback(model, callback)

        if callback.best_solution is not None:
            return callback.best_solution

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            return [v for v in nodes if solver.Value(x[v]) > 0]

        return []