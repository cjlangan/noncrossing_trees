import networkx as nx
from typing import Tuple, List, Optional

from .conflict import ConflictAnalyzer
from ..visualization import Visualizer
from ..core import TreeUtils
from ..optimization import OptimizationSolver

class Greedy:
    """Analyzer to determine the flip distance between tree pairs."""

    def __init__(self):
        self.conflict_analyzer = ConflictAnalyzer()
        self.visualizer = Visualizer()
        self.optimizer = OptimizationSolver()

    def get_flip_sequence(self, 
            T_i: List[Tuple[int, int]], 
            T_f: List[Tuple[int, int]],
            verbose: Optional[bool] = False, 
            slow: Optional[bool] = False, 
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Determines the flip sequence and distance between two NCSTs"""

        # Order edges (a, b) where a < b. For consistency
        for i in range(len(T_i)):
            a, b = sorted(T_i[i])
            c, d = sorted(T_f[i])
            T_i[i] = (a, b)
            T_f[i] = (c, d)

        curr_tree = T_i.copy() # current version of the tree

        # Create a new bipartite graph to represent edge crossings
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

        if slow: 
            Visualizer.plot_bipartite_graph(B, "bipartite_graph.png")

        sequence = [] # stores the flip sequence
        done = False

        # Main flipping loop
        while not done:
            # Sort T' edges by least crossings, then by least length
            T_f_ordered = sorted(
                (d["edge"] for n, d in B.nodes(data=True) if d["bipartite"] == 1),
                key=lambda edge: (
                    B.degree[f"f_{edge}"], 
                    TreeUtils.edge_length(edge, n)
                )
            )

            # If no remaining edge, we are done
            if len(T_f_ordered) == 0:
                break

            # Get all minima (lowest crossings)
            lowest_crossing = B.degree[f"f_{T_f_ordered[0]}"]
            # lowest_length = TreeUtils.edge_length(T_f_ordered[0], n)

            # Gath all minima (edges in T' with lowest crossings)
            minima = []
            for i in range(len(T_f_ordered)):
                if B.degree[f"f_{T_f_ordered[i]}"] == lowest_crossing:
                    # if TreeUtils.edge_length(T_f_ordered[i], n) == lowest_length:
                    minima.append(T_f_ordered[i])
            # print("Minima:", minima)

            T_i_candidate = (-1, -1) # holds candidate edge in T to flip from
            T_f_candidate = minima[0] # holds candidate edge in T' to flip into
            T_i_candidate_degree = -1 # hold degree (representing crossings) of T edge candidate
            need_park = False

            # CASE FOR DIRECT FLIP NO CROSSING
            if lowest_crossing == 0:

                if verbose:
                    print("CASE 1: FINDING EASY DIRECT FLIP")
            
                for edge in minima:
                    # Get all edge involved in cycle
                    cyc = Greedy.get_cycle_edges_after_adding(T_i_graph, edge)

                    # Prune out shared edges (assumes happy edge conjecture)
                    cyc = [(a, b) for a, b in cyc if not T_f_graph.has_edge(a, b)]

                    # check for best flip of possibilities
                    for e in cyc:
                        # We want to remove the most crossings => prioritise high degree then long length
                        # Check if new candidate crosses more, or if the same check if it is longer
                        if B.degree[f"i_{e}"] > T_i_candidate_degree or (B.degree[f"i_{e}"] == T_i_candidate_degree and TreeUtils.edge_length(e, n) > TreeUtils.edge_length(T_i_candidate, n)):
                            T_i_candidate_degree = B.degree[f"i_{e}"]
                            T_i_candidate = e
                            T_f_candidate = edge

                # need to remove T' edge if was a direct flip, which it is in this case
                B.remove_node(f"f_{T_f_candidate}")


            # CASE FOR DIRECT FLIP ONE CROSSING
            if lowest_crossing == 1:

                if verbose:
                    print("CASE 2: CHECKING FOR CROSSING DIRECT FLIP")
                
                # Find best T' candidate, and corresponding T edge (could be multiple)
                for edge in minima:
                    # Get all edge involved in cycle
                    cyc = Greedy.get_cycle_edges_after_adding(T_i_graph, edge)

                    # Prune out shared edges (assumes happy edge conjecture)
                    cyc = [(a, b) for a, b in cyc if not T_f_graph.has_edge(a, b)]

                    for e in cyc:
                        # Check if the edge can be a direct flip (also crosses)
                        if B.has_edge(f"i_{e}", f"f_{edge}"):
                            # Check if it has a higher degree, or same degree but longer
                            # COULD MAYBE ADD TIE BREAK LATER
                            if B.degree[f"i_{e}"] > T_i_candidate_degree or (B.degree[f"i_{e}"] == T_i_candidate_degree and TreeUtils.edge_length(e, n) > TreeUtils.edge_length(T_i_candidate, n)):
                                T_i_candidate_degree = B.degree[f"i_{e}"]
                                T_i_candidate = e
                                T_f_candidate = edge

                # Check if we found a direct flip
                if T_i_candidate_degree != -1:

                    # need to remove T' edge if was a direct flip, which it is in this case
                    B.remove_node(f"f_{T_f_candidate}")

                else:
                    # If no direct flip then defer to case 3
                    need_park = True


            # CASE FOR INDIRECT FLIP (to border spot)
            if lowest_crossing > 1 or need_park:
                if verbose:
                    print("CASE 3: NO DIRECT FLIPS, NEED TO PARK")

                T_i_candidate_degree = 0

                # Determine available borders (not in T)
                available_borders = []
                for i in range(n):
                    a, b = sorted((i, (i+1)%n))
                    if not T_i_graph.has_edge(a, b):
                        available_borders.append((a, b))


                highest_bord_degree = -1

                # Other variables to test condition with to choose a better flip
                # longest_cyc_length = -1 
                # highest_candidate_degree = -1

                # Find minima crosser that can be moved to border
                for bord in available_borders:

                    # Create temporary graph with border for look-ahead
                    temp = T_i_graph.copy()
                    a,b = bord
                    temp.add_edge(a, b)

                    cyc = Greedy.get_cycle_edges_after_adding(T_i_graph, bord)

                    # See if any minima neighbor is in cycle
                    for e in cyc:
                        for m in minima:
                            # If T edge is in the cycle (flippable option)
                            # AND it crosses the minima, then it is an optional flip
                            if B.has_edge(f"i_{e}", f"f_{m}"):

                                # Remove candidate from temporary graph
                                a, b = e
                                temp2 = temp.copy()
                                temp2.remove_edge(a, b)

                                # Add minima
                                a, b = m
                                temp2.add_edge(a, b)

                                # Valid if no cycle with border and minima
                                # We could test other methods of validating candidates
                                valid = not Greedy.has_cycle_with_edges(temp2, bord, m) # this has been the norm
                                # valid = True

                                # Let's try highest T' degree

                                # Get highest degree of border endpoint (actual degree; not crossings)
                                # We want higher degree to stay more connect to larger structure
                                temp3 = temp.copy()
                                temp3 = T_f_graph.copy() # Makes n=13 better, but n=12 worse
                                a,b = bord
                                temp3.add_edge(a, b)
                                bord_degree = max(temp3.degree[bord[0]], temp3.degree[bord[1]])


                                # Another variable used to maybe get a better greedy choice
                                # cyc_length = len(Greedy.get_cycle_edges_after_adding(T_f_graph, bord))


                                # Prioritise higher crossing candidate, then border degree
                                # if valid and B.degree[f"i_{e}"] > highest_candidate_degree or (B.degree[f"i_{e}"] == highest_candidate_degree and bord_degree > highest_bord_degree):

                                    # if e == (2, 11) or e == (0, 2):
                                    #     print("THIS ONE")

                                # This is the main one I've been using
                                # If valid prioritise longest candidate, THEN highest border degree
                                if valid and (TreeUtils.edge_length(e, n) > TreeUtils.edge_length(T_i_candidate, n) or (TreeUtils.edge_length(e, n) == TreeUtils.edge_length(T_i_candidate, n) and bord_degree > highest_bord_degree)):
                                                                                                                        #(cyc_length > longest_cyc_length or (cyc_length == longest_cyc_length and bord_degree > highest_bord_degree)))):

                                # Yet another method of validation
                                # if valid and B.degree[f"i_{e}"] > highest_candidate_degree:
                                    #highest_candidate_degree = B.degree[f"i_{e}"]

                                    highest_bord_degree = bord_degree
                                    T_f_candidate = bord
                                    T_i_candidate = e

                                    break

                # Since indirect, we need to add a new T edge/bipartite node
                B.add_node(f"i_{T_f_candidate}", bipartite=0, edge=T_f_candidate)


            # Add flip to sequence
            if verbose:
                print(f"Best flip: {T_i_candidate} --> {T_f_candidate}, crossing {T_i_candidate_degree} edges")
            sequence.append((T_i_candidate, T_f_candidate))

            # Execute the flip
            curr_tree.remove(T_i_candidate)
            curr_tree.append(T_f_candidate)
            a,b = T_i_candidate
            c,d = T_f_candidate
            T_i_graph.remove_edge(a,b)
            T_i_graph.add_edge(c,d)

            # Now we need to update the Bipartite graph, edge is removed so remove vertex associated
            B.remove_node(f"i_{T_i_candidate}")

            # Print trees after flip if on step by step mode
            if slow:
                input()
                Visualizer.plot_trees_together(curr_tree, T_f, "original_trees.png")

                # The rest of this just generates the conflict graph, not necessary
                # Get conflict vertices and edge pairs
                conflict_vertices, E_i, E_f = self.conflict_analyzer.get_gaps_and_edge_pairs(curr_tree, T_f)

                # Get conflict edges
                conflict_edges = self.conflict_analyzer.get_conflict_edges(
                    conflict_vertices, E_i, E_f)

                # Create conflict graph
                H = nx.DiGraph()
                H.add_nodes_from(conflict_vertices)
                for u, v, t in conflict_edges:
                    H.add_edge(u, v, type=t)

                # Find largest acyclic subgraph
                acyclic_nodes = self.optimizer.find_largest_acyclic_subgraph(H)
                ac_h = len(acyclic_nodes)
                v_h = len(conflict_vertices)
                gamma = None if v_h == 0 else ac_h / v_h

                # Feel free to add this information to output for testing
                #Visualizer.plot_conflict_graph(H, "conflict_graph.png")
                # print(f"Gamma = {gamma} = {ac_h}/{v_h}")


        if verbose:
            print("===== FLIPPING COMPLETE ======")

        return sequence


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


    @staticmethod
    def has_cycle_with_edges(G: nx.Graph, e1: Tuple[int, int], e2: Tuple[int, int]) -> bool:
        # Normalize the edges since nx.Graph is undirected
        a, b = tuple(sorted(e1))
        c, d = tuple(sorted(e2))
        e1 = (a, b)
        e2 = (c, d)

        # Check that both edges actually exist in the graph
        if not (G.has_edge(*e1) and G.has_edge(*e2)):
            return False

        # Remove e1 from G and see if e1[0] is still connected to e1[1] via a path containing e2
        G_temp = G.copy()
        G_temp.remove_edge(*e1)
        try:
            # Check all simple paths between the endpoints of e1
            for path in nx.all_simple_paths(G_temp, source=e1[0], target=e1[1]):
                # Turn path into a set of edges
                path_edges = {tuple(sorted((path[i], path[i + 1]))) for i in range(len(path) - 1)}
                if e2 in path_edges:
                    return True
        except nx.NetworkXNoPath:
            return False

        return False


    def flip_trees(self, 
            T_i: List[Tuple[int, int]], 
            T_f: List[Tuple[int, int]],
    ):
        """Tracks the transition-graph whilst flipping"""

        """
        The transition-graph is something I made up that shows where edges
        can be flipped to and the edges they conflict with (crossing currently).

        Every edge must first be represented as a vertex is the transition-graph

        Note: We include all possible edges in the graph.

        Then we draw a directed edge from e to e' if e can be flipped to e'
        """




