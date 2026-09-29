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
    
    edges = defaultdict(list)
    for _ in range(M):
        X = int(data[idx])
        idx += 1
        Y = int(data[idx])
        idx += 1
        Z = int(data[idx])
        idx += 1
        edges[X].append((Y, Z))
        edges[Y].append((X, Z))
    
    # Build graph and find connected components
    visited = [False] * (N + 1)
    components = []
    for i in range(1, N + 1):
        if not visited[i]:
            q = deque()
            q.append(i)
            visited[i] = True
            component = []
            while q:
                u = q.popleft()
                component.append(u)
                for v, z in edges[u]:
                    if not visited[v]:
                        visited[v] = True
                        q.append(v)
            components.append(component)
    
    # Assign variables to nodes in each component
    var = {}
    for comp in components:
        for node in comp:
            var[node] = len(var) + 1
    
    # Build equations
    eq = defaultdict(list)
    for X, Y, Z in zip([int(data[i]) for i in range(idx, idx + M*3, 3)], 
                      [int(data[i]) for i in range(idx + 1, idx + M*3, 3)], 
                      [int(data[i]) for i in range(idx + 2, idx + M*3, 3)]):
        u = var[X]
        v = var[Y]
        z = Z
        eq[u].append((v, z))
    
    # Solve system of equations using BFS
    A = [0] * (N + 1)
    for comp in components:
        # Choose a root
        root = comp[0]
        A[root] = 0
        q = deque()
        q.append(root)
        visited = set()
        visited.add(root)
        while q:
            u = q.popleft()
            for v, z in eq[u]:
                if v not in visited:
                    A[v] = A[u] ^ z
                    visited.add(v)
                    q.append(v)
    
    # Check for contradictions
    for X, Y, Z in zip([int(data[i]) for i in range(idx, idx + M*3, 3)], 
                      [int(data[i]) for i in range(idx + 1, idx + M*3, 3)], 
                      [int(data[i]) for i in range(idx + 2, idx + M*3, 3)]):
        if (A[X] ^ A[Y]) != Z:
            print(-1)
            return
    
    # Output the result
    print(' '.join(map(str, A[1:N+1])))

if __name__ == "__main__":
    main()
