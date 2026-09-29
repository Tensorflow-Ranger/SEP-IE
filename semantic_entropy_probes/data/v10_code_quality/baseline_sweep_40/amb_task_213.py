import sys
import collections

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    N = int(data[idx])
    idx += 1
    M = int(data[idx])
    idx += 1
    
    edges = [[] for _ in range(N + 1)]
    for _ in range(M):
        X = int(data[idx])
        idx += 1
        Y = int(data[idx])
        idx += 1
        Z = int(data[idx])
        idx += 1
        edges[X].append((Y, Z))
        edges[Y].append((X, Z))
    
    visited = [False] * (N + 1)
    A = [0] * (N + 1)
    
    def dfs(u, val):
        visited[u] = True
        A[u] = val
        for v, z in edges[u]:
            if not visited[v]:
                dfs(v, A[u] ^ z)
    
    for u in range(1, N + 1):
        if not visited[u]:
            dfs(u, 0)
    
    # Check for contradictions
    for X, Y, Z in zip([x[0] for x in edges], [y[0] for y in edges], [z[1] for x, y, z in edges]):
        if A[X] ^ A[Y] != Z:
            print(-1)
            return
    
    print(' '.join(map(str, A[1:N+1])))

if __name__ == "__main__":
    main()
