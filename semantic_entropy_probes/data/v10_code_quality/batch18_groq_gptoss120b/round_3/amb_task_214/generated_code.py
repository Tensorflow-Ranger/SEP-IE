import sys
sys.setrecursionlimit(1 << 25)

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.xor_potential = [0] * n  # xor from node to its parent representative

    def find(self, x):
        if self.parent[x] != x:
            old_parent = self.parent[x]
            self.parent[x] = self.find(self.parent[x])
            self.xor_potential[x] ^= self.xor_potential[old_parent]
        return self.parent[x]

    def union(self, x, y, xor_val):
        px = self.find(x)
        py = self.find(y)
        if px == py:
            # Check consistency
            current_xor = self.xor_potential[x] ^ self.xor_potential[y]
            return current_xor == xor_val
        else:
            # Union by rank
            if self.rank[px] < self.rank[py]:
                px, py = py, px
                x, y = y, x
            if self.rank[px] == self.rank[py]:
                self.rank[px] += 1
            self.parent[py] = px
            # Set potential such that (xor_potential[x] ^ xor_val) = xor_potential[y] relative to new root px
            # Since px is now parent of py, we want:
            # xor_potential[x] ^ xor_val ^ xor_potential[y] should hold when rooted at px
            # So: new_xor_potential[py] = xor_potential[x] ^ xor_val ^ xor_potential[y]
            # But since py's new root is px, and px's potential is 0 relative to itself,
            # we set the potential of py (relative to px) as:
            self.xor_potential[py] = self.xor_potential[x] ^ xor_val ^ self.xor_potential[y]
            return True

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    N = int(data[idx]); idx += 1
    M = int(data[idx]); idx += 1
    
    if M == 0:
        # No constraints, all zeros work and minimize sum
        print(' '.join(['0']*N))
        return
        
    edges = []
    for _ in range(M):
        x = int(data[idx]) - 1; idx += 1
        y = int(data[idx]) - 1; idx += 1
        z = int(data[idx]); idx += 1
        edges.append((x, y, z))
        
    uf = UnionFind(N)
    for x, y, z in edges:
        if not uf.union(x, y, z):
            print(-1)
            return
            
    # Now extract assignments
    # For each component, we can assign the root to 0, and compute others based on potentials
    result = [0] * N
    for i in range(N):
        root = uf.find(i)
        # A[i] = xor_potential[i] (since root is assigned 0)
        result[i] = uf.xor_potential[i]
        
    print(' '.join(map(str, result)))

if __name__ == "__main__":
    main()
