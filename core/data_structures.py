class UnionFind:
    """Union-Find data structure for cycle detection."""

    def __init__(self):
        self.parent = {}

    def __getitem__(self, x):
        if x not in self.parent:
            self.parent[x] = x
        if self.parent[x] != x:
            self.parent[x] = self[self.parent[x]]
        return self.parent[x]

    def union(self, x, y):
        px, py = self[x], self[y]
        self.parent[px] = py
