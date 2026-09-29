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
    
    def dfs(u, parent, value):
        visited[u] = True
        A[u] = value
        for v, z in edges[u]:
            if v == parent:
                continue
            if visited[v]:
                if (A[u] ^ A[v]) != z:
                    return False
            else:
                if not dfs(v, u, A[u] ^ z):
                    return False
        return True
    
    for u in range(1, N + 1):
        if not visited[u]:
            if not dfs(u, -1, 0):
                print(-1)
                return
    
    print(' '.join(map(str, A[1:N+1])))

if __name__ == "__main__":
    main()
