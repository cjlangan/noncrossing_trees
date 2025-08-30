from ..analysis import GammaAnalyzer
from ..core import TreeUtils
from ..generation import NCSTGenerator

def best_trees_demo():
    gamma_analyzer = GammaAnalyzer()

    print("""Which best pair do you want to test?
        0: Trees from Bjerkevik's Paper
        1: Symmetric n=13 pair
        2: n=12 pair""")
    choice = int(input())

    if choice == 0:
        # Trees from the paper, n = 13
        T_i = [(0,3),(0,4),(1,3),(4,7),(4,8),(5,7),(8,11),(8,12),(9,11),(2, 3),(6,7),(10,11)]
        T_f = [(0,10),(1,10),(1,5),(2,4),(2,5),(5,9),(6,8),(6,9),(10,12),(2,3),(6,7),(10,11)]
    elif choice == 1:
        # New symmetric pair, n = 13
        T_i = [(0,3),(3,12),(1,3),(4,7),(4,8),(5,7),(8,11),(8,12),(9,11),(2, 3),(6,7),(10,11)]
        T_f = [(0,10),(1,10),(1,5),(2,4),(2,5),(5,9),(6,8),(6,9),(10,12),(2,3),(6,7),(10,11)]
    elif choice == 2:
        # This is the best pair of trees (lowest known gamma with the lowest number of vertices)
        T_i = [(5,9),(0,9),(1,4),(0,11),(1,5),(7,8),(2,4),(3,4),(6,8),(5,8),(0,10)]
        T_f = [(6,2),(11,2),(10,7),(11,0),(10,6),(4,3),(9,7),(8,7),(5,3),(6,3),(11,1)]
    elif choice == 3:
        T_i = [(0,4),(1,4),(1,3),(1,2),(4,15),(5,15),(5,10),(6,10),(6,9),(6,8),(6,7),
               (10,14),(11,14),(11,13),(11,12),(15,20),(16,20),(16,19),(16,18),(16,17)]
        T_f = [(0,2),(1,2),(2,19),(2,20),(3,8),(3,7),(4,7),(5,7),(6,7),(8,19),(8,18),
               (9,13),(9,12),(10,12),(11,12),(13,18),(13,17),(14,17),(15,17),(16,17)]
    else:
        # Default to symmetric n=13 pair
        T_i = [(0,3),(3,12),(1,3),(4,7),(4,8),(5,7),(8,11),(8,12),(9,11),(2, 3),(6,7),(10,11)]
        T_f = [(0,10),(1,10),(1,5),(2,4),(2,5),(5,9),(6,8),(6,9),(10,12),(2,3),(6,7),(10,11)]

    print("=== Analyzing Best Known Tree Pair (gamma ≈ 0.44444444) ===")
    gamma_analyzer.analyze_tree_pair(T_i, T_f, verbose=True, plot=True)

if __name__ == "__main__":
    best_trees_demo()
