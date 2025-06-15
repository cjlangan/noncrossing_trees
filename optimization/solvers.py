import pulp
import networkx as nx
from typing import List


class OptimizationSolver:
    """Solver for optimization problems on conflict graphs."""

    @staticmethod
    def find_largest_acyclic_subgraph(H: nx.DiGraph) -> List[int]:
        """Find the largest acyclic subgraph using integer programming."""
        model = pulp.LpProblem("MaxAcyclicSubgraph", pulp.LpMaximize)

        # Decision variables
        x = {v: pulp.LpVariable(f"x_{v}", cat="Binary") for v in H.nodes}
        order = {v: pulp.LpVariable(f"order_{v}", lowBound=0,
                                    upBound=len(H.nodes)-1, cat="Integer")
                 for v in H.nodes}

        # Objective: maximize included nodes
        model += pulp.lpSum(x[v] for v in H.nodes)

        # Acyclicity constraints
        M = len(H.nodes)
        for u, v in H.edges():
            model += order[u] + 1 <= order[v] + M * (2 - x[u] - x[v])

        # Solve quietly
        model.solve(pulp.PULP_CBC_CMD(msg=False))

        return [v for v in H.nodes if x[v].varValue == 1]
