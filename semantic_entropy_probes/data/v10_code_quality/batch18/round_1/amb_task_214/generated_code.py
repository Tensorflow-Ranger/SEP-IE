import sys
import collections

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
    
    # Build graph
    graph = collections.defaultdict(list)
    for x, y, z in edges:
        graph[x].append((y, z))
        graph[y].append((x, z))
    
    # Assign variables to nodes
    # We'll use BFS to assign values starting from node 1
    # We'll use a dictionary to store the values of each node
    A = [0] * (N + 1)  # 1-based indexing
    visited = [False] * (N + 1)
    queue = collections.deque()
    
    # Start from node 1
    queue.append(1)
    visited[1] = True
    A[1] = 0  # We can choose any value for the starting node
    
    # BFS to assign values
    while queue:
        u = queue.popleft()
        for v, z in graph[u]:
            if not visited[v]:
                visited[v] = True
                A[v] = A[u] ^ z
                queue.append(v)
            else:
                # Check if the existing value is consistent
                if A[v] != A[u] ^ z:
                    print(-1)
                    return
    
    # Check for cycles and consistency
    # We can do a second pass to verify all constraints
    for x, y, z in edges:
        if A[x] ^ A[y] != z:
            print(-1)
            return
    
    # The sum is minimized by setting the starting node to 0 and propagating
    # Since we can choose any value for the starting node, setting it to 0 gives the minimal sum
    print(' '.join(map(str, A[1:N+1])))

if __name__ == "__main__":
    main()
