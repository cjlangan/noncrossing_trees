from ..enumeration import FiniteGammaSearcher

def enumeration_search_demo():
    finite_searcher = FiniteGammaSearcher()

    finite_searcher.enumerate_ncsts_k_borders_parallel(
            10,     # vertices
            3,      # number of border edges
            test=True   # option of wether to test or just enumerate
    )
        

if __name__ == "__main__":
    enumeration_search_demo()
