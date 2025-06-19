from ..enumeration import FiniteGammaSearcher

def restricted_enumeration_demo():
    even_searcher = FiniteGammaSearcher()
    even_searcher.enumerate_even_shared_full(12, 4)

if __name__ == "__main__":
    restricted_enumeration_demo()
