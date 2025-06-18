import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
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
        """Generate conflict graph visualization."""
        edge_color_map = {1: 'black', 2: 'purple', 3: 'orange'}
        edge_colors = [edge_color_map[H[u][v]['type']] for u, v in H.edges()]
        edge_labels = nx.get_edge_attributes(H, 'type')

        pos = nx.shell_layout(H)

        plt.figure(figsize=(6, 6), dpi=300)
        nx.draw(H, pos, with_labels=True, node_color='lightblue',
                edge_color=edge_colors, node_size=2000, arrows=True)
        nx.draw_networkx_edge_labels(H, pos, edge_labels=edge_labels)

        # Create legend
        legend_elements = [
            Line2D([0], [0], color='black', lw=2, label='Type 1'),
            Line2D([0], [0], color='purple', lw=2, label='Type 2'),
            Line2D([0], [0], color='orange', lw=2, label='Type 3')
        ]
        plt.legend(handles=legend_elements, loc='upper left')
        plt.title("Conflict Directed Graph with Edge Type Legend")

        if filename:
            plt.savefig(f"./{PROJECT_NAME}/{filename}", bbox_inches='tight')
            plt.close()
            print(f"Generated conflict graph as {filename}")
        else:
            plt.show()


    @staticmethod
    def print_tree(T, filename=None):
        """Generate an image of a single tree in convex position"""
        n = len(T) + 1
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
