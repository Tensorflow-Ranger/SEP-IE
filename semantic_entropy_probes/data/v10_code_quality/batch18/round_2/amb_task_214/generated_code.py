import sys
import sys
from collections import defaultdict, deque

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    N = int(data[idx])
    idx += 1
    M = int(data[idx])
    idx += 1
    
    edges = []
    for _ in range(M):
        x = int(data[idx])
        idx += 1
        y = int(data[idx])
        idx += 1
        z = int(data[idx])
        idx += 1
        edges.append((x, y, z))
    
    # Build graph and track XOR constraints
    graph = defaultdict(list)
    for x, y, z in edges:
        graph[x].append((y, z))
        graph[y].append((x, z))
    
    # Union-Find to detect contradictions
    parent = list(range(N + 1))
    rank = [1] * (N + 1)
    
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    
    def union(u, v):
        pu = find(u)
        pv = find(v)
        if pu == pv:
            return False
        if rank[pu] < rank[pv]:
            parent[pu] = pv
        else:
            parent[pv] = pu
            if rank[pu] == rank[pv]:
                rank[pu] += 1
        return True
    
    # Check for contradictions using Union-Find
    for x, y, z in edges:
        if find(x) == find(y):
            # Already in the same component, check if the XOR is consistent
            # We need to track the XOR from root to each node
            pass
    
    # Build a bipartition for each component using BFS
    # We'll use a dictionary to store the XOR from root to each node
    xor_from_root = {}
    
    visited = [False] * (N + 1)
    for node in range(1, N + 1):
        if not visited[node]:
            # BFS to assign values
            q = deque()
            q.append(node)
            visited[node] = True
            xor_from_root[node] = 0
            while q:
                u = q.popleft()
                for v, z in graph[u]:
                    if not visited[v]:
                        visited[v] = True
                        xor_from_root[v] = xor_from_root[u] ^ z
                        q.append(v)
    
    # Now, assign A[i] = xor_from_root[i]
    # But we need to check for contradictions
    # For each edge (x, y, z), check if A[x] ^ A[y] == z
    # If any contradiction, output -1
    
    A = [0] * (N + 1)
    for i in range(1, N + 1):
        A[i] = xor_from_root[i]
    
    for x, y, z in edges:
        if (A[x] ^ A[y]) != z:
            print(-1)
            return
    
    # Output the result
    print(' '.join(map(str, A[1:N+1])))

if __name__ == "__main__":
    main()
