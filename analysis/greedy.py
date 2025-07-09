import networkx as nx
from typing import Tuple, List

from .conflict import ConflictAnalyzer
from ..visualization import Visualizer
from ..core import TreeUtils

class Greedy:
    """Analyzer to determine the flip distance between tree pairs."""

    def __init__(self):
        self.conflict_analyzer = ConflictAnalyzer()
        self.visualizer = Visualizer()


    def get_flip_sequence(self, 
            T_i: List[Tuple[int, int]], 
            T_f: List[Tuple[int, int]]
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Determines the flip sequence and distance between two NCSTs"""

        self.visualizer.plot_trees_together(T_i, T_f, "original_trees.png")
        self.visualizer.plot_linear_graph(T_i, T_f, T_i, T_f, "linear_graph.png")
        # self.visualizer.plot_bipartite_graph(B, "bipartite_graph.png")

        curr_Ti = T_i.copy()

        # Create a new bipartite graph
        B = nx.Graph()
        n = len(T_i) + 1

        # Create graph of T
        T_i_graph = nx.Graph()
        T_i_graph.add_nodes_from(list(range(n)))
        T_i_graph.add_edges_from(T_i)

        # Create graph of T'
        T_f_graph = nx.Graph()
        T_f_graph.add_nodes_from(list(range(n)))
        T_f_graph.add_edges_from(T_f)


        # Add vertices that represent edges in T and T'
        nodes_Ti = []
        for edge in T_i:
            if edge not in T_f:
                nodes_Ti.append(edge)
                label = f"i_{edge}"
                B.add_node(label, bipartite=0, edge=edge)

        nodes_Tf = []
        for edge in T_f:
            if edge not in T_i:
                nodes_Tf.append(edge)
                label = f"f_{edge}"
                B.add_node(label, bipartite=1, edge=edge)

        # Add edges between nodes in different partitions if the represented edges cross
        for ei in T_i:
            for ef in T_f:
                if TreeUtils.cross(ei, ef):
                    B.add_edge(f"i_{ei}", f"f_{ef}")


        # Flip Loop
        for i in range(20):
            # Sort T' edges by least crossings, then by least length
            T_f_ordered = sorted(
                (d["edge"] for n, d in B.nodes(data=True) if d["bipartite"] == 1),
                key=lambda edge: (
                    B.degree[f"f_{edge}"], 
                    TreeUtils.edge_length(edge, n)
                )
            )

            # Get all minima (lowest crossings and lowest length)
            lowest_crossing = B.degree[f"f_{T_f_ordered[0]}"]
            lowest_length = TreeUtils.edge_length(T_f_ordered[0], n)

            if lowest_crossing != 0:
                print("LOWEST CROSSING NOT ZERO")
                break

            # CASE FOR WHEN LOWEST CROSSING IS ZERO
            
            minima = []
            for i in range(len(T_f_ordered)):
                if B.degree[f"f_{T_f_ordered[i]}"] == lowest_crossing:
                    if TreeUtils.edge_length(T_f_ordered[i], n) == lowest_length:
                        minima.append(T_f_ordered[i])
            print("Minima:", minima)


            T_i_candidate_degree = -1
            T_i_candidate = (-1, -1)
            T_f_candidate = minima[0]
            all_cyc = set()

            for edge in minima:
                # Get all edge involved in cycle
                cyc = Greedy.get_cycle_edges_after_adding(T_i_graph, edge)

                # Prune out shared edges (assumes happy edge conjecture)
                cyc = [(a, b) for a, b in cyc if not T_f_graph.has_edge(a, b)]

                # Add options to all candidates and check for best flip
                for e in cyc:
                    all_cyc.add(e)

                    # We want to remove the most crossings => prioritise high degree then long length
                    # Check if new candidate crosses more, or if the same check if it is longer
                    if B.degree[f"i_{e}"] > T_i_candidate_degree or (B.degree[f"i_{e}"] == T_i_candidate_degree and TreeUtils.edge_length(e, n) > TreeUtils.edge_length(T_i_candidate, n)):
                        T_i_candidate_degree = B.degree[f"i_{e}"]
                        T_i_candidate = e
                        T_f_candidate = edge

                # print(f"Options to flip into {edge} are {cyc}")

            print(f"Best flip: {T_i_candidate} --> {T_f_candidate}, crossing {T_i_candidate_degree} edges")

            print("Executing flip...")
            curr_Ti.remove(T_i_candidate)
            curr_Ti.append(T_f_candidate)
            a,b = T_i_candidate
            c,d = T_f_candidate
            T_i_graph.remove_edge(a,b)
            T_i_graph.add_edge(c,d)

            # Print trees after flip
            input()
            Visualizer.plot_trees_together(curr_Ti, T_f, "original_trees.png")

            # Now we need to update the Bipartite graph, edge is removed so remove vertex associated
            # Also need to remove T' edge if was a direct flip, which it is in this case
            B.remove_node(f"i_{T_i_candidate}")
            B.remove_node(f"f_{T_f_candidate}")


        # CASE FOR ZERO COMPLETE
        # NOW NEED TO DO THE CASE FOR ONLY ONES
        # NEED TO CHECK IF THERE ARE ANY DIRECT FLIPS
        # OF DIRECT FLIPS, CHOOSE THE ONE WITH MOST CROSSINGS THE LONGEST LENGTH
        # IN CASE OF TIE BREAKING, CHOOSE SHORTEST T_f EDGE

        # THEN NEED TO DO THE COMPLICATED CASE FOR WHEN NEED TO PARK AN EDGE
        

        # If empty, choose connecting chord that is not a leaf with most crossings to flip into it
        #   - further, must choose red edges that would form cycle with it. (should only ever be one cycle)
        #   - does one always exist? yes, since putting edge there would make a cycle, and 
        #       we can then pick any other edge in the cycle to put there.
        #   - Prioritise: high crossings, chords
        #
        # So if empty then there is always an edge to flip into it:
        #   - take the edge in the cycle with the most crossings
        #
        # In general for multiple empty, prioritise the ones on the borders
        # 
        # Eventually will have the border complete.
        #
        # If one crossing, it is possible that it can be flipped in, also possible that it cannot.
        #
        # SEE WHITEBOARD FOR IDEAS ON THIS ALGORITHM
        # IMPLEMENTATION WILL BE QUITE DIFFICULT (AT LEAST IN ENSURING GOOD TIME COMPLEXITY)
        # BUT AS LONG AS THE LOGIC IS THERE THEN WE CAN CONSTRUCT IT AND THEN BEGIN TESTING
        # AGAINST THE KNOWN BEST MINMUM FLIP SEQUENCES.

        return []


    @staticmethod
    def get_cycle_edges_after_adding(
        G: nx.Graph, 
        edge: Tuple[int, int], 
    ) -> List[Tuple[int, int]]:
        """
        Given a tree G and an edge (a, b) that is added to it,
        return all other edges in the cycle formed,
        with edges in the form (min, max).
        """
        a,b = edge
        if not nx.has_path(G, a, b):
            return []

        # Find unique path from a to b
        path = nx.shortest_path(G, source=a, target=b)

        # Convert the path to normalized edges (a <= b)
        cycle_edges = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            cycle_edges.append((min(u, v), max(u, v)))

        return cycle_edges
