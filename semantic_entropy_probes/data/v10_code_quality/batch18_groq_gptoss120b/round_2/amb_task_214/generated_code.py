import sys
from collections import defaultdict

sys.setrecursionlimit(1 << 25)

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    N = int(data[idx])
    idx += 1
    M = int(data[idx])
    idx += 1
    
    if M == 0:
        # No constraints, all zeros is optimal
        print(' '.join(['0']*N))
        return

    edges = []
    for _ in range(M):
        x = int(data[idx]) - 1  # Convert to 0-indexed
        idx += 1
        y = int(data[idx]) - 1
        idx += 1
        z = int(data[idx])
        idx += 1
        edges.append((x, y, z))

    # Union-Find to manage connected components
    parent = list(range(N))
    def find(u):
        if parent[u] != u:
            parent[u] = find(parent[u])
        return parent[u]
    
    def union(u, v):
        pu, pv = find(u), find(v)
        if pu != pv:
            parent[pu] = pv

    # Build Union-Find structure based on edges
    for x, y, z in edges:
        union(x, y)

    # Group edges by component
    components = defaultdict(list)
    for x, y, z in edges:
        root = find(x)
        components[root].append((x, y, z))

    # Initialize result array
    A = [None] * N

    # Process each component
    for root, edge_list in components.items():
        # Collect all nodes in this component
        nodes_set = set()
        for x, y, z in edge_list:
            nodes_set.add(x)
            nodes_set.add(y)
        nodes = list(nodes_set)
        node_index = {node: i for i, node in enumerate(nodes)}
        n_nodes = len(nodes)

        # Start assigning values from node 0 in component
        # Set first node to 0
        start_node = nodes[0]
        A[start_node] = 0
        visited = [False] * n_nodes
        visited[node_index[start_node]] = True
        queue = [start_node]

        # BFS to assign values
        while queue:
            u = queue.pop()
            u_idx = node_index[u]
            for x, y, z in edge_list:
                if x == u:
                    v = y
                elif y == u:
                    v = x
                else:
                    continue
                v_idx = node_index[v]
                required = A[u] ^ z
                if A[v] is None:
                    A[v] = required
                    visited[v_idx] = True
                    queue.append(v)
                else:
                    if A[v] != required:
                        print(-1)
                        return

        # Check if all nodes in component were visited/assigned
        for i in range(n_nodes):
            if not visited[i]:
                A[nodes[i]] = 0  # Assign zero to unvisited (isolated) nodes in component

    # Assign 0 to any remaining unassigned nodes (components with no edges)
    for i in range(N):
        if A[i] is None:
            A[i] = 0

    print(' '.join(map(str, A)))

if __name__ == "__main__":
    main()
