import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
import networkx as nx
from typing import List, Tuple, Optional

PROJECT_NAME = "noncrossing_trees"

class Visualizer:
    """Class for generating visualizations of trees and graphs."""

    @staticmethod
    def print_progress_bar(current: int, total: int, bar_length: int = 40):
        """Print a progress bar."""
        percent = current / total if total > 0 else 0
        filled_len = int(bar_length * percent)
        bar = '=' * filled_len + '-' * (bar_length - filled_len)
        sys.stdout.write(f"\rProgress: [{bar}] {percent:.1%} ({current}/{total})")
        sys.stdout.flush()

    @staticmethod
    def plot_trees_together(T: List[Tuple[int, int]],
                            T_prime: List[Tuple[int, int]],
                            filename: Optional[str] = None):
        """Plot two trees together on convex position."""
        n = len(T) + 1
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        points = np.stack([np.cos(angles), np.sin(angles)], axis=1)

        fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
        ax.set_aspect('equal')
        ax.axis('off')

        # Draw convex polygon
        for i in range(n):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % n]
            ax.plot([x1, x2], [y1, y2], color='gray',
                    linestyle='--', linewidth=0.5)

        # Categorize edges
        T_set = {tuple(sorted(e)) for e in T}
        T_prime_set = {tuple(sorted(e)) for e in T_prime}
        common_edges = T_set & T_prime_set
        only_T = T_set - common_edges
        only_Tp = T_prime_set - common_edges

        # Draw edges with different colors
        edge_groups = [
            (only_T, 'red', 'T'),
            (only_Tp, 'blue', "T'"),
            (common_edges, 'purple', 'common')
        ]

        for edges, color, label in edge_groups:
            for (i, j) in edges:
                xi, yi = points[i]
                xj, yj = points[j]
                ax.plot([xi, xj], [yi, yj], color=color,
                        linewidth=2, alpha=0.7, label=label)

        # Create legend without duplicates
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), loc='upper right')

        # Label vertices
        for i, (x, y) in enumerate(points):
            ax.text(x * 1.08, y * 1.08, str(i),
                    fontsize=12, ha='center', va='center')

        plt.title("Original NCSTs on Shared Vertices")

        if filename:
            plt.savefig(f"./{PROJECT_NAME}/{filename}", bbox_inches='tight')
            plt.close(fig)
            print(f"Generated shared tree graph as {filename}")
        else:
            plt.show()

    @staticmethod
    def plot_linear_graph(T_i: List[Tuple[int, int]],
                          T_f: List[Tuple[int, int]],
                          E_i: List[Tuple[int, int]],
                          E_f: List[Tuple[int, int]],
                          filename: Optional[str] = None):
        """Generate linear representation of trees."""
        n = len(T_i) + 1
        x = np.arange(n)
        y = np.zeros(n)

        fig, ax = plt.subplots(figsize=(max(6, n * 0.6), 8), dpi=300)

        # Plot vertices
        ax.scatter(x, y, color='black', zorder=5)
        for i in range(n):
            ax.text(x[i] - 0.1, y[i], str(i),
                    ha='right', va='center', fontsize=8)

        def draw_arc(a: int, b: int, color: str, above: bool = True, bold: bool = False):
            if a == b:
                return
            a, b = sorted((a, b))
            center = (x[a] + x[b]) / 2
            width = abs(x[b] - x[a])
            height = width * 5
            theta1, theta2 = (0, 180) if above else (180, 0)
            lw = 2.5 if bold else 1.0
            arc = Arc((float(center), 0), width=float(width), height=float(height),
                      angle=0, theta1=theta1, theta2=theta2, color=color, linewidth=lw)
            ax.add_patch(arc)

        # Draw tree edges
        for (a, b) in T_i:
            draw_arc(a, b, color='red', above=True,
                     bold=(a, b) in E_i or (b, a) in E_i)

        for (a, b) in T_f:
            draw_arc(a, b, color='blue', above=False,
                     bold=(a, b) in E_f or (b, a) in E_f)

        # Set limits
        vertical_margin = n * 3
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(-vertical_margin, vertical_margin)
        ax.axis('off')

        plt.title("Linear Tree Representation", fontsize=10)

        # Add legend
        legend_elements = [
            Line2D([0], [0], color='red', lw=1, label="T (initial tree)"),
            Line2D([0], [0], color='blue', lw=1, label="T' (final tree)"),
            Line2D([0], [0], color='gray', lw=2.5,
                   label="Near-near pair", alpha=0.8)
        ]
        ax.legend(handles=legend_elements, loc='upper left',
                  fontsize=8, frameon=False)

        if filename:
            plt.savefig(f"./{PROJECT_NAME}/{filename}", bbox_inches='tight')
            plt.close(fig)
            print(f"Generated linear graph as {filename}")
        else:
            plt.show()


    @staticmethod
    def plot_conflict_graph(H: nx.DiGraph, filename: Optional[str] = None):
        """Generate conflict graph visualization with curved edges to avoid overlap."""
        edge_color_map = {1: 'black', 2: 'purple', 3: 'orange'}

        # Extract edges and colors
        edges = list(H.edges(data=True))
        edge_colors = [edge_color_map[d['type']] for _, _, d in edges]
        edgelist = [(u, v) for u, v, _ in edges]

        # Circular layout (nodes on convex hull of a circle)
        pos = nx.circular_layout(H)

        plt.figure(figsize=(6, 6), dpi=300)

        # Draw nodes (smaller)
        nx.draw_networkx_nodes(H, pos, node_color='lightblue', node_size=800)

        # Draw edges with slight curvature
        for (u, v), color in zip(edgelist, edge_colors):
            start, end = Visualizer.shorten_edge(pos, u, v, node_radius=0.08)  # match node size
            arrow = FancyArrowPatch(start, end, # type: ignore[arg-type]
                            arrowstyle='-|>',
                            color=color,
                            linewidth=2,
                            mutation_scale=15,
                            connectionstyle="arc3,rad=0.25",
                            zorder=1)
            plt.gca().add_patch(arrow)

        # Draw node labels only
        nx.draw_networkx_labels(H, pos, font_size=10, font_weight="bold")

        # Remove axes/black box
        plt.axis("off")

        # Legend
        legend_elements = [
            Line2D([0], [0], color='black', lw=2, label='Type 1'),
            Line2D([0], [0], color='purple', lw=2, label='Type 2'),
            Line2D([0], [0], color='orange', lw=2, label='Type 3')
        ]
        plt.legend(handles=legend_elements, loc='upper left')
        plt.title("Conflict Graph")

        if filename:
            plt.savefig(f"./{PROJECT_NAME}/{filename}", bbox_inches='tight')
            plt.close()
            print(f"Generated conflict graph as {filename}")
        else:
            plt.show()


    @staticmethod
    def plot_bipartite_graph(B: nx.Graph, filename: Optional[str] = None):
        # Separate nodes by bipartite attribute
        nodes_Ti = [n for n, d in B.nodes(data=True) if d["bipartite"] == 0]
        nodes_Tf = [n for n, d in B.nodes(data=True) if d["bipartite"] == 1]

        # Adjust height based on number of nodes
        max_nodes = max(len(nodes_Ti), len(nodes_Tf))
        height_per_node = 0.7
        fig_height = max(4, height_per_node * max_nodes)
        fig, ax = plt.subplots(figsize=(10, fig_height), dpi=200)

        # Layout
        pos = nx.bipartite_layout(B, nodes_Ti)

        # Draw nodes
        nx.draw_networkx_nodes(
            B, pos, nodelist=nodes_Ti, node_color='white',
            edgecolors='red', linewidths=2.0, node_size=1500, label="T", ax=ax
        )
        nx.draw_networkx_nodes(
            B, pos, nodelist=nodes_Tf, node_color='white',
            edgecolors='blue', linewidths=2.0, node_size=1500, label="T'", ax=ax
        )

        # Optional: show edge tuple as node label
        labels = {n: f"{B.nodes[n]['edge']}" for n in B.nodes}

        nx.draw_networkx_edges(B, pos, ax=ax)
        nx.draw_networkx_labels(B, pos, labels=labels, font_size=8, ax=ax)

        ax.axis('off')
        ax.set_title("Bipartite Graph of Crossing Edges", pad=40)
        ax.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, 1.02),
            ncol=2,
            frameon=False,
            handletextpad=2.0
        )

        plt.tight_layout()

        if filename:
            plt.savefig(f"./{PROJECT_NAME}/{filename}", bbox_inches='tight', dpi=300)
            plt.close(fig)
            print(f"Generated bipartite graph as {filename}")
        else:
            plt.show()

    @staticmethod
    def plot_tree(T: List[Tuple[int, int]], filename: Optional[str] = None):
        """Plot a single tree together in convex position."""
        n = len(T) + 1
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        points = np.stack([np.cos(angles), np.sin(angles)], axis=1)

        fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
        ax.set_aspect('equal')
        ax.axis('off')

        # Draw convex polygon
        for i in range(n):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % n]
            ax.plot([x1, x2], [y1, y2], color='gray',
                    linestyle='--', linewidth=0.5)

        for (i, j) in T:
            xi, yi = points[i]
            xj, yj = points[j]
            ax.plot([xi, xj], [yi, yj], color='black',
                    linewidth=2, alpha=0.7)

        # Label vertices
        for i, (x, y) in enumerate(points):
            ax.text(x * 1.08, y * 1.08, str(i),
                    fontsize=12, ha='center', va='center')

        if filename:
            plt.savefig(f"./{PROJECT_NAME}/{filename}", bbox_inches='tight')
            plt.close(fig)
            print(f"Generated NCST as {filename}")
        else:
            plt.show()

    @staticmethod
    def shorten_edge(pos, u, v, node_radius=0.05):
        """Return modified start/end positions so arrow stops at node boundary."""
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        vec = np.array([x2 - x1, y2 - y1])
        length = np.linalg.norm(vec)
        if length == 0:
            return (x1, y1), (x2, y2)
        # Shorten the vector by node_radius at target
        vec_unit = vec / length
        new_end = np.array([x2, y2]) - vec_unit * node_radius
        return (x1, y1), new_end
