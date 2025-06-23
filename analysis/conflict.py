import networkx as nx
from typing import List, Tuple



class ConflictAnalyzer:
    """Analyzer for conflict graphs and edge relationships."""

    @staticmethod
    def edge_length(edge: Tuple[int, int]) -> int:
        """Compute edge length in linear representation."""
        a, b = edge
        return abs(a - b)

    @staticmethod
    def is_near_edge(edge: Tuple[int, int], idx: int) -> bool:
        """Check if edge is a near edge for given gap index."""
        a, b = sorted(edge)
        return ((a == idx and b > idx + 1) or
                (b == idx + 1 and a < idx))

    @staticmethod
    def get_gaps_and_edge_pairs(
        T_i: List[Tuple[int, int]],
        T_f: List[Tuple[int, int]]
    ) -> Tuple[List[int], List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Determine conflict gaps and associated edge pairs."""
        num_gaps = len(T_i)
        conflict_vertices = []
        T_i_associated = [None] * num_gaps
        T_f_associated = [None] * num_gaps

        # Find shortest near edges for each gap
        for i in range(num_gaps):
            for T, associated in [(T_i, T_i_associated), (T_f, T_f_associated)]:
                shortest_near = None

                for a, b in T:
                    if i not in (a, b) or i+1 not in (a, b):  # Not a short edge
                        if ConflictAnalyzer.is_near_edge((a, b), i):
                            if (shortest_near is None or
                                    ConflictAnalyzer.edge_length((a, b))
                                    < ConflictAnalyzer.edge_length(
                                    shortest_near
                                    )):
                                shortest_near = (a, b)

                associated[i] = shortest_near  # type: ignore

        # Remove near edges where short edges exist
        for T, associated in [(T_i, T_i_associated), (T_f, T_f_associated)]:
            for a, b in T:
                if abs(a - b) == 1:
                    i = min(a, b)
                    associated[i] = None

        # Identify conflict vertices (near-near pairs)
        for i in range(num_gaps):
            if T_i_associated[i] is not None and T_f_associated[i] is not None:
                conflict_vertices.append(i)
            else:
                T_i_associated[i] = None
                T_f_associated[i] = None

        # Filter out None values
        T_i_edges = [edge for edge in T_i_associated if edge is not None]
        T_f_edges = [edge for edge in T_f_associated if edge is not None]

        return conflict_vertices, T_i_edges, T_f_edges  # type: ignore

    @staticmethod
    def get_conflict_edges(
        conflict_vertices: List[int],
        E_i: List[Tuple[int, int]],
        E_f: List[Tuple[int, int]]
    ) -> List[Tuple[int, int, int]]:
        """Get conflict edges with their types."""
        conflict_edges = []

        for i, (a, b) in enumerate(E_i):
            for j, (c, d) in enumerate(E_f):
                if i != j:
                    a, b = sorted((a, b))
                    c, d = sorted((c, d))
                    g_i, g_j = conflict_vertices[i], conflict_vertices[j]

                    # Type 1: crossing edges
                    if a < c < b < d or c < a < d < b:
                        conflict_edges.append((g_i, g_j, 1))
                    # Type 2: e_j' covers e_i and e_i covers g_j
                    elif c <= a and b <= d and a <= g_j and g_j+1 <= b:
                        conflict_edges.append((g_i, g_j, 2))
                    # Type 3: e_i covers e_j' and e_j' covers g_i
                    elif a <= c and d <= b and c <= g_i and g_i+1 <= d:
                        conflict_edges.append((g_i, g_j, 3))

        return conflict_edges

    @staticmethod
    def get_conflict_graph(
        conflict_vertices: List[int],
        conflict_edges: List[Tuple[int, int, int]],
    )-> nx.Graph:
        """Construct the conflict graph from edges and vertices."""

        H = nx.Graph()
        H.add_nodes_from(conflict_vertices)

        for u, v, edge_type in conflict_edges:
            H.add_edge(u, v, type=edge_type)

        return H