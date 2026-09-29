import sys
from collections import defaultdict

def find_parent(parent, x):
    if parent[x] != x:
        parent[x] = find_parent(parent, parent[x])
    return parent[x]

def union(parent, rank, x, y):
    px, py = find_parent(parent, x), find_parent(parent, y)
    if px == py:
        return
    if rank[px] < rank[py]:
        px, py = py, px
    parent[py] = px
    if rank[px] == rank[py]:
        rank[px] += 1

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    if M == 0:
        print(' '.join(['0'] * N))
        return
    
    idx = 2
    edges = []
    for _ in range(M):
        x = int(data[idx]) - 1
        y = int(data[idx+1]) - 1
        z = int(data[idx+2])
        edges.append((x, y, z))
        idx += 3

    parent = list(range(N))
    rank = [0] * N
    xor_val = [0] * N

    for x, y, z in edges:
        px = find_parent(parent, x)
        py = find_parent(parent, y)
        
        if px == py:
            if xor_val[x] ^ xor_val[y] != z:
                print(-1)
                return
        else:
            union(parent, rank, px, py)
            xor_val[py] = xor_val[x] ^ xor_val[y] ^ z

    res = [0] * N
    for i in range(N):
        root = find_parent(parent, i)
        res[i] = xor_val[i] ^ xor_val[root]
    
    print(' '.join(map(str, res)))

if __name__ == "__main__":
    main()
