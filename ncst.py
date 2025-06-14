# Author: Connor Langan
# Institution: University of Manitoba
#
# Description:
#   This file contains multiple different functions to test non-crossing spanning trees
#   in convex position. This includes generation graphs, testing functions for gamma, 
#   random gamma searching, and NCST enumeration with fixed number of border edges.
#
#   You can start by uncommenting and commenting out lines in the main method to test
#   these different function to get a feel for what they do. You may also find the 
#   function headers useful.

import networkx as nx
import numpy as np
import pulp
import random
import secrets
import sys
from multiprocessing import Process, Event, Manager, Lock
import multiprocessing
import matplotlib.pyplot as plt
from networkx.utils import UnionFind
from matplotlib.lines import Line2D
from matplotlib.patches import Arc
from itertools import combinations
from more_itertools import distinct_permutations
from math import gcd, comb
from sympy import divisors, totient
from tqdm import tqdm
from itertools import islice


# Generates an image of the two trees on the same points in convex position
def print_trees_together(T, T_prime, filename=None):
    n = len(T) + 1
    # Generate n points on a unit circle
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    points = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    ax.set_aspect('equal')
    ax.axis('off')

    # Draw the convex polygon
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        ax.plot([x1, x2], [y1, y2], color='gray', linestyle='--', linewidth=0.5)

    # Convert edge sets to sets of sorted tuples
    T_set = {tuple(sorted(e)) for e in T}
    T_prime_set = {tuple(sorted(e)) for e in T_prime}
    common_edges = T_set & T_prime_set
    only_T = T_set - common_edges
    only_Tp = T_prime_set - common_edges

    def draw_edges(edges, color, label):
        for (i, j) in edges:
            xi, yi = points[i]
            xj, yj = points[j]
            ax.plot([xi, xj], [yi, yj], color=color, linewidth=2, alpha=0.7, label=label)

    draw_edges(only_T, 'red', 'T')
    draw_edges(only_Tp, 'blue', "T'")
    draw_edges(common_edges, 'purple', 'common')

    # Avoid duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc='upper right')

    # Label the nodes
    for i, (x, y) in enumerate(points):
        ax.text(x * 1.08, y * 1.08, str(i), fontsize=12, ha='center', va='center')

    plt.title("Original NCSTs on Shared Vertices")

    if filename:
        plt.savefig(filename, bbox_inches='tight')
        plt.close(fig)
        print("Generated shared tree graph as", filename)
    else:
        plt.show()


# Generate an image of a single tree in convex position
def print_tree(T, filename=None):
    n = len(T) + 1  # A tree with n-1 edges has n vertices
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    points = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    ax.set_aspect('equal')
    ax.axis('off')

    # Draw the convex polygon outline
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        ax.plot([x1, x2], [y1, y2], color='gray', linestyle='--', linewidth=0.5)

    # Draw the tree edges
    for (i, j) in T:
        xi, yi = points[i]
        xj, yj = points[j]
        ax.plot([xi, xj], [yi, yj], color='black', linewidth=2, alpha=0.8)

    # Label the vertices
    for i, (x, y) in enumerate(points):
        ax.text(x * 1.08, y * 1.08, str(i), fontsize=12, ha='center', va='center')

    plt.title("Tree on Convex Polygon")

    if filename:
        plt.savefig(filename, bbox_inches='tight')
        plt.close(fig)
        print("Generated tree graph as", filename)
    else:
        plt.show()


# Generate Linear Representation of Trees
def print_linear_graph(T_i, T_f, E_i, E_f, filename=None):
    n = len(T_i) + 1
    x = np.arange(n)
    y = np.zeros(n)

    # Smaller width, taller height to make arcs higher and nodes closer horizontally
    fig, ax = plt.subplots(figsize=(max(6, n * 0.6), 8), dpi=300)  # width reduced, height increased
    
    # Plot vertices closer horizontally
    ax.scatter(x, y, color='black', zorder=5)
    for i in range(n):
        ax.text(x[i] - 0.1, y[i], str(i), ha='right', va='center', fontsize=8)

    def draw_arc(a, b, color, above=True, bold=False):
        if a == b:
            return
        a, b = sorted((a, b))
        center = (x[a] + x[b]) / 2
        width = abs(x[b] - x[a])
        height = width * 5  # Increased height multiplier for higher arcs
        theta1, theta2 = (0, 180) if above else (180, 0)
        lw = 2.5 if bold else 1.0
        arc = Arc((float(center), 0), width=float(width), height=float(height), angle=0, theta1=theta1, theta2=theta2, color=color, linewidth=lw)
        ax.add_patch(arc)

    # Draw T_i edges (red, above)
    for (a, b) in T_i:
        draw_arc(a, b, color='red', above=True, bold=(a, b) in E_i or (b, a) in E_i)

    # Draw T_f edges (blue, below)
    for (a, b) in T_f:
        draw_arc(a, b, color='blue', above=False, bold=(a, b) in E_f or (b, a) in E_f)

    # Set limits with vertical margin to capture all arcs fully
    vertical_margin = n * 3  # increase to fit arcs height
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-vertical_margin, vertical_margin)

    ax.axis('off')

    plt.title("Linear Tree Representation", fontsize=10)

    legend_elements = [
        Line2D([0], [0], color='red', lw=1, label="T (initial tree)"),
        Line2D([0], [0], color='blue', lw=1, label="T' (final tree)"),
        Line2D([0], [0], color='gray', lw=2.5, label="Near-near pair", alpha=0.8)
    ]

    ax.legend(handles=legend_elements, loc='upper left', fontsize=8, frameon=False)

    if filename:
        plt.savefig(filename, bbox_inches='tight')
        plt.close(fig)
        print("Generated linear graph as", filename)
    else:
        plt.show()


# Generates an image of the conflict graph
def print_conflict_graph(H, filename=None):
    edge_color_map = {1: 'black', 2: 'purple', 3: 'orange'}
    edge_colors = [edge_color_map[H[u][v]['type']] for u, v in H.edges()]
    edge_labels = nx.get_edge_attributes(H, 'type')

    # Shell layout for a circular appearance
    pos = nx.shell_layout(H)

    # Draw the graph
    plt.figure(figsize=(6, 6), dpi=300)
    nx.draw(
        H, pos,
        with_labels=True,
        node_color='lightblue',
        edge_color=edge_colors,
        node_size=2000,
        arrows=True
    )
    nx.draw_networkx_edge_labels(H, pos, edge_labels=edge_labels)

    # Create custom legend
    legend_elements = [
        Line2D([0], [0], color='black', lw=2, label='Type 1'),
        Line2D([0], [0], color='purple', lw=2, label='Type 2'),
        Line2D([0], [0], color='orange', lw=2, label='Type 3')
    ]
    plt.legend(handles=legend_elements, loc='upper left')

    plt.title("Conflict Directed Graph with Edge Type Legend")

    # Save or show
    if filename:
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print("Generated conflict graph as", filename)
    else:
        plt.show()

def binomial(n, k):
    """Compute binomial coefficient C(n, k) = n! / (k! * (n-k)!)"""
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    
    # Use the multiplicative formula for efficiency
    result = 1
    for i in range(min(k, n - k)):
        result = result * (n - i) // (i + 1)
    return result

def U(n, k):
    """
    Compute U(n, k) = (2/(n-2)) * C(n-2, k) * sum_{j=0}^{k-1} C(n-1, j) * C(n-k-2, k-1-j) * 2^{n-1-2k+j}
    """
    if k < 0 or n <= 2 or k >= n:
        return 0
    
    # First term: 2/(n-2) * C(n-2, k)
    coeff = 2 / (n - 2)
    binom_term = binomial(n - 2, k)
    
    # Sum term
    sum_value = 0
    for j in range(k):  # j from 0 to k-1
        term1 = binomial(n - 1, j)
        term2 = binomial(n - k - 2, k - 1 - j)
        exp_term = 2**(n - 1 - 2*k + j)
        sum_value += term1 * term2 * exp_term
    
    return coeff * binom_term * sum_value

# Function that computes the number of NCSTs on n vertices with exactly k borders
def T(n, k):
    """
    Compute T(n, k) = U(n, k-1) - U(n, k) + C(n-1, k) * (1/(n-1)) * sum_{j=0}^{k-1} C(n-1, j) * C(n-k-1, k-1-j) * 2^{n-2k+j}
    """
    if k < 2 or k >= n or n <= 2:
        return 0
    
    # First two terms: U(n, k-1) - U(n, k)
    u_diff = U(n, k - 1) - U(n, k)
    
    # Third term: C(n-1, k) * (1/(n-1)) * sum
    coeff = binomial(n - 1, k) / (n - 1)
    
    # Sum term
    sum_value = 0
    for j in range(k):  # j from 0 to k-1
        term1 = binomial(n - 1, j)
        term2 = binomial(n - k - 1, k - 1 - j)
        exp_term = 2**(n - 2*k + j)
        sum_value += term1 * term2 * exp_term
    
    additional_term = coeff * sum_value
    
    return (int)(u_diff + additional_term)



# Counts number of unique k borders on n convex vertices
def count_binary_necklaces(n, k):
    """
    Calculate the number of binary necklaces of length n with exactly k ones.
    """
    if k < 0 or k > n:
        return 0

    g = gcd(n, k)
    total = 0

    for d in divisors(g):
        phi_d = totient(d)
        binom = comb(n // d, k // d)
        total += phi_d * binom

    return total // n

def rotate(lst, k):
    return lst[k:] + lst[:k]

def min_rotation(bitstring):
    """Find lex smallest rotation of the bitstring."""
    return min(rotate(bitstring, i) for i in range(len(bitstring)))

# Iterates over unique border combinations up to rotational isomorphism, now
# with the additional option for up to rotation and reflective isomorphism
def generate_binary_necklaces(n, k, reflective=False):
    """Generate binary necklaces of length n with k ones.
    If reflective=True, only yield up to flip (dihedral) symmetry.
    """
    seen = set()
    base = [1] * k + [0] * (n - k)

    for perm in distinct_permutations(base):
        canon = tuple(min_rotation(perm))

        if reflective:
            flipped = tuple(min_rotation(perm[::-1]))
            canon = min(canon, flipped)

        if canon not in seen:
            seen.add(canon)
            yield canon

# Function I haven't done much testing with yet, supposed to ensure that
# the border edges are evenly spaced.
def evenly_spaced_border_combination(n, k):
    if k == 0:
        return []
    if n % k != 0:
        return "Not possible: k does not divide n"

    step = n // k
    border_edges = [((i * step) % n, ((i * step + 1) % n)) for i in range(k)]
    return border_edges


def print_progress_bar(current, total, bar_length=40):
    percent = current / total
    filled_len = int(bar_length * percent)
    bar = '=' * filled_len + '-' * (bar_length - filled_len)
    sys.stdout.write(f"\rProgress: [{bar}] {percent:.1%} ({current}/{total})")
    sys.stdout.flush()


# Function to enumerate over all NCSTs with exactly k borders
#   PARAMETERS:
#       n: number of vertices
#       k: number of borders
#       test: if False, then just enumerates without testing for gammas
#
# NOTE: Currently this function only test a found tree against its flip counterpart.
#       That is, it does NOT test every conceivable combination.
def enumerate_ncsts_k_borders(n, k, test=True):
    seen = set()
    num_tested = 0
    best_gamma = 1
    best_trees = [], []
    four_nine_gamma_counter = 0
    four_nine_list = []

    print(f"Beginning test on all NCSTSs with {n} vertices and {k} borders")
    num_trees = T(n, k)
    print_progress_bar(num_tested, num_trees)

    def is_valid_tree(points, edges):
        # A graph is a tree iff it is connected and has n-1 edges
        if len(edges) != len(points) - 1:
            return False
        G = nx.Graph()
        G.add_nodes_from(points)
        G.add_edges_from(edges)
        return nx.is_connected(G)

    def has_cycle_uf(edges):
        uf = UnionFind()
        for u, v in edges:
            if uf[u] == uf[v]:
                return True
            uf.union(u, v)
        return False

    def enumerate_ncsts_helper(points, local_edges, all_edges):
        nonlocal num_tested, seen, best_gamma, best_trees, four_nine_gamma_counter


        if len(all_edges) == n - 1 and is_valid_tree(range(n), all_edges):
            base_tree = [(min(a, b), max(a, b)) for a, b in all_edges]
            flipped_tree = flip_tree(base_tree)

            def rotated_versions(tree):
                return [
                    frozenset(
                        (min((a + r) % n, (b + r) % n), max((a + r) % n, (b + r) % n))
                        for a, b in tree
                    )
                    for r in range(n)
                ]

            for tree_variant in (base_tree, flipped_tree):
                for rotated in rotated_versions(tree_variant):
                    if rotated in seen:
                        continue
                    seen.add(rotated)
                    num_tested += 1

                    gamma = None
                    if test:
                        gamma, ac_h, E_i, E_f, H = test_trees(
                            rotated, flip_tree(rotated), verbose=False, plot=False
                        )

                    print_progress_bar(num_tested, num_trees)

                    if test and gamma is not None:
                        if gamma < best_gamma:
                            best_gamma = gamma
                            best_trees = [sorted(rotated), sorted(flip_tree(rotated))]

                        if gamma <= 4 / 9:
                            four_nine_gamma_counter += 1
                            four_nine_list.append(rotated)

                        if gamma < 4 / 9:
                            print("\n✅✅✅FOUND BETTER GAMMA✅✅✅", flush=True)
                            print(f"Gamma = {gamma}", flush=True)
                            print(f"Used {sorted(rotated)} and flip", flush=True)
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

                new_edge = (a, b)
                new_local_edges = local_edges + [new_edge]
                new_all_edges = all_edges | {edge}

                if has_cycle_uf(new_local_edges):
                    continue

                a_idx, b_idx = sorted((points.index(a), points.index(b)))
                between = points[a_idx + 1:b_idx]
                outside = points[:a_idx + 1] + points[b_idx:]

                between_edges = [e for e in new_local_edges if e[0] in between and e[1] in between]
                outside_edges = [e for e in new_local_edges if e[0] in outside and e[1] in outside]

                enumerate_ncsts_helper(between, between_edges, new_all_edges)
                enumerate_ncsts_helper(outside, outside_edges, new_all_edges)

    for necklace in generate_binary_necklaces(n, k, reflective=False):
        borders = [(i, (i + 1) % n) for i, b in enumerate(necklace) if b == 1]
        border_set = {tuple(sorted(e)) for e in borders}
        enumerate_ncsts_helper(list(range(n)), borders, border_set)

    print(f"\nTesting complete. Total NCSTSs tested: {num_tested}")
    print(f"Best gamma: {best_gamma}")
    print(f"Found on trees: {best_trees[0]} and {best_trees[1]}")
    print(f"Number of 4/9 or better: {four_nine_gamma_counter}")
    print(f"All 4/9 or better: {four_nine_list}")


# Global variables for shared state (will be initialized in main process)
shared_counter = None
total_trees = None

def init_worker(counter, total):
    """Initialize worker process with shared variables"""
    global shared_counter, total_trees
    shared_counter = counter
    total_trees = total

def chunkify(iterable, n):
    """Split iterable into n chunks as evenly as possible."""
    items = list(iterable)
    k, m = divmod(len(items), n)
    return (items[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def worker_process(n, k, necklace_batch, test):
    global shared_counter, total_trees
    seen = set()
    local_best_gamma = 1
    local_best_trees = [], []
    local_four_nine_gamma_counter = 0
    local_four_nine_list = []
    local_num_tested = 0

    def is_valid_tree(points, edges):
        if len(edges) != len(points) - 1:
            return False
        G = nx.Graph()
        G.add_nodes_from(points)
        G.add_edges_from(edges)
        return nx.is_connected(G)

    def has_cycle_uf(edges):
        uf = UnionFind()
        for u, v in edges:
            if uf[u] == uf[v]:
                return True
            uf.union(u, v)
        return False

    def enumerate_ncsts_helper(points, local_edges, all_edges):
        nonlocal local_num_tested, seen, local_best_gamma, local_best_trees, local_four_nine_gamma_counter


        if len(all_edges) == n - 1 and is_valid_tree(range(n), all_edges):
            base_tree = [(min(a, b), max(a, b)) for a, b in all_edges]
            flipped_tree = flip_tree(base_tree)

            def rotated_versions(tree):
                return [
                    frozenset(
                        (min((a + r) % n, (b + r) % n), max((a + r) % n, (b + r) % n))
                        for a, b in tree
                    )
                    for r in range(n)
                ]

            for tree_variant in (base_tree, flipped_tree):
                for rotated in rotated_versions(tree_variant):
                    if rotated in seen:
                        continue
                    seen.add(rotated)
                    local_num_tested += 1

                    # Update shared counter and print progress
                    with shared_counter.get_lock():
                        shared_counter.value += 1
                        if shared_counter.value % 10 == 0:
                            print_progress_bar(shared_counter.value, total_trees)

                    gamma = None
                    if test:
                        gamma, *_ = test_trees(rotated, flip_tree(rotated), verbose=False, plot=False)

                    if test and gamma is not None:
                        if gamma < local_best_gamma:
                            local_best_gamma = gamma
                            local_best_trees = [sorted(rotated), sorted(flip_tree(rotated))]
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
                if has_cycle_uf(new_local_edges):
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
        border_set = {tuple(sorted(e)) for e in borders}
        enumerate_ncsts_helper(list(range(n)), borders, border_set)

    return local_num_tested, local_best_gamma, local_best_trees, local_four_nine_gamma_counter, local_four_nine_list


# Define a separate function that can be pickled
def worker_wrapper(args):
    n, k, necklace_batch, test = args
    return worker_process(n, k, necklace_batch, test)


# Function to enumerate over all NCSTs with exactly k borders
#   PARAMETERS:
#       n: number of vertices
#       k: number of borders
#       test: if False, then just enumerates without testing for gammas
#
# NOTE: Currently this function only test a found tree against its flip counterpart.
#       That is, it does NOT test every conceivable combination.
def enumerate_ncsts_k_borders_parallel(n, k, test=True):
    print(f"Parallel NCST Search for n={n}, k={k}")
    total_trees_expected = T(n, k)  # Use the proper T(n,k) function
    print(f"Expected total trees: {total_trees_expected}")
    
    all_necklaces = list(generate_binary_necklaces(n, k, reflective=True))

    cpu_count = multiprocessing.cpu_count()
    necklace_chunks = list(chunkify(all_necklaces, cpu_count))
    
    # Create shared counter for progress tracking
    counter = multiprocessing.Value('i', 0)

    # Create argument tuples for the worker function
    worker_args = [(n, k, chunk, test) for chunk in necklace_chunks]
    
    print_progress_bar(0, total_trees_expected)

    with multiprocessing.Pool(cpu_count, initializer=init_worker, initargs=(counter, total_trees_expected)) as pool:
        results = list(pool.map(worker_wrapper, worker_args))
        
    # Final progress update
    with counter.get_lock():
        print_progress_bar(counter.value, counter.value)  # Show 100%
    print()  # New line after progress bar

    total_tested = 0
    best_gamma = 1
    best_trees = [], []
    total_four_nine = 0
    all_four_nine = []

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


# Determine the length of an edge in linear representation
def edge_length(edge):
    a,b = edge
    return abs(a-b)


# Determine if an edge is a near edge
def is_near_edge(edge, idx):
    a,b = sorted(edge)
    is_near = False

    # Check that the left index appears once and reaches right
    if a == idx and b > idx + 1:
        is_near = True

    # Else check that the right index appears once and reaches left
    elif b == idx + 1 and a < idx:
        is_near = True

    return is_near


# Function to return a rotated tree by k places
def rotate_tree(T, k):
    n = len(T) + 1
    rotated_tree = []

    for a, b in T:
        a_rot = (a + k) % n
        b_rot = (b + k) % n
        rotated_tree.append((a_rot, b_rot))

    return rotated_tree


# Function to return the flipped tree
def flip_tree(T):
    n = len(T) + 1
    flipped_tree = []

    for a, b in T:
        a_flipped = (n - a) % n
        b_flipped = (n - b) % n
        flipped_tree.append((a_flipped, b_flipped))

    return rotate_tree(flipped_tree, -1)


# Function to blowup a tree
#   PARAMETERS:
#       T: the tree
#       k: number of added edges per gap
#       E: T's near edges from near-near pairs
#       conflict_vertices: the conflict graphs conflict vertices
def blowup_tree(T, k, E, conflict_vertices):
    T_blown = list(T)
    all_vertices = set()

    for i in range(len(E)):
        a, b = sorted(E[i])
        cv = conflict_vertices[i] # the conflict vertex

        # Evenly space new vertices within the gap
        new_fractions = np.linspace(cv, cv+1, k + 2)[1:-1]
        all_vertices.update(new_fractions)
            
        # Determine endpoint
        endpoint = b if a == cv else a

        # Add new edges to tree
        for v in new_fractions:
            T_blown.append((v, endpoint))

    # Normalize all vertices: assign integer labels based on sorted order
    for u, v in T_blown:
        all_vertices.add(u)
        all_vertices.add(v)

    sorted_vertices = sorted(all_vertices)
    mapping = {v: i for i, v in enumerate(sorted_vertices)}

    def normalize(v):
        return mapping[v]

    T_normalized = [(normalize(u), normalize(v)) for (u, v) in T_blown]

    return T_normalized


# Function to determine the conflict gaps and associated edge pairs
def get_gaps_and_edge_pairs(T_i, T_f):
    num_gaps = len(T_i)
    conflict_vertices = []
    T_i_associated_edges = [None] * num_gaps
    T_f_associated_edges = [None] * num_gaps

    # For each possible gap, we check if it is one to keep
    for i in range(num_gaps):
        shortest_near = None  # candidate edge to associate

        # Check all edges in T_i
        for a,b in T_i:
            # Ensure the edge is not a short edge
            if i not in (a,b) or i+1 not in (a,b):
                if is_near_edge((a,b), i):
                    # If near edge check if shortest
                    if shortest_near is None or edge_length((a,b)) < edge_length(shortest_near):
                        shortest_near = (a, b)

        # Add best candidate to the associated edge array
        T_i_associated_edges[i] = shortest_near

        shortest_near = None  # candidate edge to associate

        # Check all edges in T_f
        for a,b in T_f:
            # Ensure the edge is not a short edge
            if i not in (a,b) or i+1 not in (a,b):
                if is_near_edge((a,b), i):
                    # If near edge check if shortest
                    if shortest_near is None or edge_length((a,b)) < edge_length(shortest_near):
                        shortest_near = (a, b)

        # Add best candidate to the associated edge array
        T_f_associated_edges[i] = shortest_near

    # If a gap has a short edge in T_i, take away the best near edge
    for a,b in T_i:
        if abs(a-b) == 1:
            i = min(a, b)
            T_i_associated_edges[i] = None

    # If a gap has a short edge in T_f, take away the best near edge
    for a,b in T_f:
        if abs(a-b) == 1:
            i = min(a, b)
            T_f_associated_edges[i] = None

    # Determine Conflict Vertices and prune non near-near pairs
    for i in range(num_gaps):
        if T_i_associated_edges[i] is not None and T_f_associated_edges[i] is not None:
            # If a gap a near-near edge pair, add it to the set of conflict vertices
            conflict_vertices.append(i)
        else:
            # Otherwise, set the associated edges to None
            T_i_associated_edges[i] = None
            T_f_associated_edges[i] = None

    # Remove None values and collapse
    T_i_associated_edges = [edge for edge in T_i_associated_edges if edge is not None]
    T_f_associated_edges = [edge for edge in T_f_associated_edges if edge is not None]

    return conflict_vertices, T_i_associated_edges, T_f_associated_edges


# Function to return and label the conflict edges
def get_conflict_edges(conflict_vertices, E_i, E_f):
    conflict_edges = []

    # Check all pairs of edges
    for i, (a, b) in enumerate(E_i):
        for j, (c, d) in enumerate(E_f):
            # Edges cannot conflict with themsleves
            if i != j:
                # Sort for convenience
                a,b = sorted((a,b))
                c,d = sorted((c,d))

                # Convert indices to conflict vertices
                g_i = conflict_vertices[i]
                g_j = conflict_vertices[j]

                # Check for Type 1 edges, cross if vertices alternate
                if a < c < b < d or c < a < d < b:
                    # Add edge from g_i to g_j
                    conflict_edges.append((g_i, g_j, 1))
                # Check for Type 2 edges, e_j' covers e_i and e_i covers g_j
                elif c <= a and b <= d and a <= g_j and g_j+1 <= b: 
                    conflict_edges.append((g_i, g_j, 2))
                # Check for Type 3 edges, e_i covers e_j' and e_j' covers g_i
                elif a <= c and d <= b and c <= g_i and g_i+1 <= d:
                    conflict_edges.append((g_i, g_j, 3))

    return conflict_edges


# Function to return the largest acycle subgraph
def find_largest_acyclic_subgraph(H):
    model = pulp.LpProblem("MaxAcyclicSubgraph", pulp.LpMaximize)

    # Binary decision variable: include node or not
    x = {v: pulp.LpVariable("x_{}".format(v), cat="Binary") for v in H.nodes}

    # Order variable for topological sorting
    order = {v: pulp.LpVariable("order_{}".format(v), lowBound=0, upBound=len(H.nodes)-1, cat="Integer") for v in H.nodes}

    # Objective: Maximize included nodes
    model += pulp.lpSum(x[v] for v in H.nodes)

    # Acyclicity constraints
    M = len(H.nodes)
    for u, v in H.edges():
        model += order[u] + 1 <= order[v] + M * (2 - x[u] - x[v])

    # Queit Solve
    model.solve(pulp.PULP_CBC_CMD(msg=False))

    return [v for v in H.nodes if x[v].varValue == 1]


# Function to generate and return a random NCST
def get_ncst(n, seed=None):
    if seed is None:
        seed = secrets.randbits(32)

    random.seed(seed)
    np.random.seed(seed)

    parent = list(range(n))
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    def union(u, v):
        parent[find(u)] = find(v)

    edges = []
    all_possible_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_possible_edges)

    for u, v in all_possible_edges:
        if find(u) != find(v):
            # Sort endpoints so u < v
            u, v = sorted((u, v))
            valid = True

            for a, b in edges:
                a, b = sorted((a, b))

                # Ensure edges don't share any vertex
                if len({u, v, a, b}) == 4:
                    # Check for crossing: vertices alternate
                    if (u < a < v < b) or (a < u < b < v):
                        valid = False
                        break

            if valid:
                edges.append((u, v))
                union(u, v)

        if len(edges) == n - 1:
            break

    return edges, seed


def is_hull_edge(u, v, n):
    # Hull edges are between consecutive vertices in circular order
    return (abs(u - v) == 1) or (u == 0 and v == n - 1) or (u == n - 1 and v == 0)

def count_border_edges(edges, n):
    return sum(1 for u, v in edges if is_hull_edge(u, v, n))

def get_ncst_with_k_border_edges(n, k, max_tries=100_000):
    for _ in range(max_tries):
        edges, seed = get_ncst(n)
        num_border = count_border_edges(edges, n)
        if num_border == k:
            return edges, seed
    raise RuntimeError(f"No NCST with exactly {k} border edges found after {max_tries} tries.")


# Function to test the k blowup of two trees
def test_blowup_trees(T_i, T_f, k, verbose=True, plot=True):
    # Get the vertices of H = H(T,T') AND the near-near edge pairs associated
    conflict_vertices, E_i, E_f = get_gaps_and_edge_pairs(T_i, T_f)

    # Get the k blowups
    T_i_blown = blowup_tree(T_i, k, E_i, conflict_vertices)
    T_f_blown = blowup_tree(T_f, k, E_f, conflict_vertices)

    test_trees(T_i_blown, T_f_blown, verbose, plot)


# Function to test two specific input trees
def test_trees(T_i, T_f, verbose=True, plot=True):
    n = len(T_i) + 1
    
    # Get the vertices of H = H(T,T') AND the near-near edge pairs associated
    conflict_vertices, E_i, E_f = get_gaps_and_edge_pairs(T_i, T_f)

    # Next we get the edges and edge type for the Conflict Graph
    conflict_edges = get_conflict_edges(conflict_vertices, E_i, E_f)

    # Then we can create the conflict graph
    H = nx.DiGraph()
    H.add_nodes_from(conflict_vertices)
   
    for u,v,t in conflict_edges:
        H.add_edge(u, v, type=t)

    # Find the largest acyclic subgraph of the conflict graph
    acyclic_nodes = find_largest_acyclic_subgraph(H)
    ac_h = len(acyclic_nodes)
    v_h = len(conflict_vertices)
    gamma = None if v_h == 0 else ac_h / v_h
    
    if verbose:
        print(f"Found gamma of: {gamma} = {ac_h}/{v_h} on {n} vertices.")
        print("Nodes included in the subgraph:", acyclic_nodes)
    
    if gamma is not None and gamma < 4/9:
        print("✅✅✅ FOUND BETTER PAIR OF TREES ✅✅✅")
        print(T_i)
        print(T_f)
    
    if plot:
        print_trees_together(T_i, T_f, "original_trees.png")
        print_linear_graph(T_i, T_f, E_i, E_f, "linear_graph.png")
        print_conflict_graph(H, "conflict_graph.png")

    return gamma, ac_h, E_i, E_f, H


# Function to trial find NCST on n vertices, until find specified gamma
# Methods to get second tree:
#   f: flip
#   r: rotate
#   fr: flip then rotate
#   rf: rotate then flip
#   random
def find_trees_with_gamma(n, gamma, method="random", k=None, verbose=True, notable=True, plot=True, skip_half=False, stop_event=None, result_holder=None, lock=None):
    curr_gamma = ac_h = v_h = 99
    seed_i = seed_f = 0
    T_i = T_f = E_i = E_f = []
    H = nx.DiGraph()
    num_tested = 0

    while (curr_gamma is None or curr_gamma > gamma) and (stop_event is None or not stop_event.is_set()):
        if k is None:
            T_i, seed_i = get_ncst(n)
        else:
            T_i, seed_i = get_ncst_with_k_border_edges(n, k)

        num_tested += 1
        seed_f = method

        if method == "rf":
            T_f = flip_tree(rotate_tree(T_i, int(n/2)))
        elif method == "fr":
            T_f = rotate_tree(flip_tree(T_i), int(n/2))
        elif method == "f":
            T_f = flip_tree(T_i)
        elif method == "r":
            T_f = rotate_tree(T_i, int(n/2))
        else:
            if k is None:
                T_f, seed_f = get_ncst(n)
            else:
                T_f, seed_f = get_ncst_with_k_border_edges(n, k)

        curr_gamma, ac_h, E_i, E_f, H = test_trees(T_i, T_f, verbose=False, plot=False)
        v_h = len(H.nodes)

        # Optionally ignore 0.5s
        if skip_half and curr_gamma == 0.5:
            curr_gamma = 1

        # Print gamma if notable
        if curr_gamma is not None and (curr_gamma < gamma or verbose or (notable and curr_gamma < 0.6)):
            print(f"Found gamma of: {curr_gamma} = {ac_h}/{v_h} on {n} vertices with seeds {seed_i} and {seed_f}.")

    if verbose:
        print("Number of NCST pairs tested:", num_tested)

    # If result_holder is used, populate and set event
    if result_holder is not None:
        if lock is not None:
            with lock:
                result_holder['total_tested'] += num_tested

                if curr_gamma is not None and curr_gamma <= gamma and stop_event is not None and not stop_event.is_set():
                    result_holder.update({
                        'curr_gamma': curr_gamma,
                        'T_i': T_i,
                        'T_f': T_f,
                        'seed_i': seed_i,
                        'seed_f': seed_f,
                        'H': H,
                        'E_i': E_i,
                        'E_f': E_f
                    })
                    stop_event.set()

    # Plot final result if interactive
    if plot and curr_gamma is not None and curr_gamma <= gamma:
        print_trees_together(T_i, T_f, "original_trees.png")
        print_linear_graph(T_i, T_f, E_i, E_f, "linear_graph.png")
        print_conflict_graph(H, "conflict_graph.png")


# Function to parallelize finding a pair of NCSTs on n vertices under a specified gamma value
# Methods to get second tree:
#   f: flip
#   r: rotate
#   fr: flip then rotate
#   rf: rotate then flip
#   random
#
# Parameters:
#   n: number of vertices
#   gamma_threshold: Gamma trying to find, or better
#   method: method to generate second NSCT (see above options)
#   k: number of border edges, default is it doesn't matter
#   num_workers: number of parallel processes to run
#   notable: whether to print notable gammas or not
#   plot: whether you want to generate graphs
def find_trees_with_gamma_parallel(n, gamma_threshold, method="random", k=None, num_workers=2, notable=True, plot=True):
    manager = Manager()
    stop_event = Event()
    result_holder = manager.dict()
    result_holder['total_tested'] = 0

    lock = Lock()

    print(f"Beginning search for gamma of {gamma_threshold} on {n} vertices...")

    workers = []
    for _ in range(num_workers):
        p = Process(target=find_trees_with_gamma, args=(n, gamma_threshold, method, k, False, notable, False, True, stop_event, result_holder, lock))
        p.start()
        workers.append(p)

    for p in workers:
        p.join()

    if 'curr_gamma' in result_holder:
        print(f"Found gamma of: {result_holder['curr_gamma']} = {len(result_holder['E_i'])}/{len(result_holder['H'].nodes)} on {n} vertices with seeds {result_holder['seed_i']} and {result_holder['seed_f']}.")
        print("Number of NCST pairs tested:", result_holder['total_tested'])

        if plot:
            print_trees_together(result_holder['T_i'], result_holder['T_f'], "original_trees.png")
            print_linear_graph(result_holder['T_i'], result_holder['T_f'], result_holder['E_i'], result_holder['E_f'], "linear_graph.png")
            print_conflict_graph(result_holder['H'], "conflict_graph.png")
    else:
        print("No gamma found meeting the threshold")


if __name__ == "__main__":
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
    # T_i = get_ncst(17, 355920981)[0]
    # T_f = flip_tree(T_i)

    # Trees I found with gamma = .44444444
    # T_i = [(8, 14), (3, 15), (9, 11), (3, 8), (9, 13), (0, 15), (5, 8), (5, 7), (4, 8), (2, 15), (9, 12), (1, 15), (10, 11), (13, 14), (5, 6)]
    # T_f = [(7, 1), (12, 0), (6, 4), (12, 7), (6, 2), (15, 0), (10, 7), (10, 8), (11, 7), (13, 0), (6, 3), (14, 0), (5, 4), (2, 1), (10, 9)]

    # Trees I found with gamma = .44444444, n = 12
    # This is the best pair of trees (lowest known gamma with the lowest number of vertices)
    T_i = [(5, 9), (0, 9), (1, 4), (0, 11), (1, 5), (7, 8), (2, 4), (3, 4), (6, 8), (5, 8), (0, 10)]
    T_f = [(6, 2), (11, 2), (10, 7), (11, 0), (10, 6), (4, 3), (9, 7), (8, 7), (5, 3), (6, 3), (11, 1)]

    # Best tree with added cell, no good
    # T_i = [(5, 9), (0, 13), (1, 4), (0, 15), (1, 5), (7, 8), (2, 4), (3, 4), (6, 8), (5, 8), (0, 14), (9, 13), (9, 12), (10, 12), (11, 12)]
    # T_f = flip_tree(T_i)

    # Nam Trees with edge removal
    # T_i = [(3, 4), (10, 11), (5, 6), (2, 15), (4, 6), (8, 14), (0, 15), (1, 15), (9, 14), (3, 7), (2, 8), (3, 8), (9, 12), (10, 12), (9, 13)]
    # T_f = [(12, 11), (5, 4), (10, 9), (13, 0), (11, 9), (7, 1), (15, 0), (14, 0), (6, 1), (12, 8), (13, 7), (12, 7), (6, 3), (5, 3), (6, 2)]

    test_trees(T_i, T_f)

    # test_trees(rotate_tree(T_i, 1), rotate_tree(T_f, 1))

    # test_blowup_trees(T_i, T_f, 3)

    # find_trees_with_gamma_parallel(12, .5, method="f", k=3, num_workers=6)

    # enumerate_ncsts_k_borders(10, 3, test=True)

    # enumerate_ncsts_k_borders_parallel(12, 3, test=True)

