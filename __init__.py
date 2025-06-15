"""
Non-Crossing Spanning Trees (NCST) Analysis Framework

A modular, class-based framework for analyzing NCSTs, their properties,
and computing gamma values for tree pairs.

In addition, we also support generating a family of graphs obtained from a given conflict graph.
"""

from .core import MathUtils, TreeUtils, UnionFind
from .generation import NCSTGenerator, NecklaceGenerator
from .analysis import GammaAnalyzer, ConflictAnalyzer, ParallelGammaSearcher
from .formulas import NCSTFormulas
from .optimization import OptimizationSolver
from .visualization import Visualizer

__version__ = "0.1.0"
__author__ = ["Connor", "Atish"]

__all__ = [
    'MathUtils',
    'TreeUtils',
    'UnionFind',
    'NCSTGenerator',
    'NecklaceGenerator',
    'GammaAnalyzer',
    'ConflictAnalyzer',
    'ParallelGammaSearcher',
    'NCSTFormulas',
    'OptimizationSolver',
    'Visualizer'
]
