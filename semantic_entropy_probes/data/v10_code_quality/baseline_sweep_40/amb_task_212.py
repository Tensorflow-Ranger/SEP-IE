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
    parent = [0] * (N + 1)
    value = [0] * (N + 1)
    is_bipartite = True
    components = []
    
    for start in range(1, N + 1):
        if not visited[start]:
            queue = collections.deque()
            queue.append(start)
            visited[start] = True
            color = {start: 0}
            component = []
            while queue:
                u = queue.popleft()
                component.append(u)
                for v, z in edges[u]:
                    if not visited[v]:
                        visited[v] = True
                        color[v] = color[u] ^ 1
                        parent[v] = u
                        value[v] = value[u] ^ z
                        queue.append(v)
                    else:
                        if color[v] == color[u]:
                            is_bipartite = False
            components.append(component)
    
    if not is_bipartite:
        print(-1)
        return
    
    for i in range(1, N + 1):
        if parent[i] == 0 and i != 1:
            value[i] = 0
    
    print(' '.join(map(str, value[1:N+1])))

if __name__ == "__main__":
    main()
