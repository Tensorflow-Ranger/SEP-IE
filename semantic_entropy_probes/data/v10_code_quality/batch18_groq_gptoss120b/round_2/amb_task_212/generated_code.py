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
    
    edges = []
    index = 2
    for _ in range(M):
        x = int(data[index]) - 1
        y = int(data[index+1]) - 1
        z = int(data[index+2])
        edges.append((x, y, z))
        index += 3
    
    # Union-Find to group variables
    parent = list(range(N))
    rank = [0] * N
    
    # First pass: build connected components
    for x, y, z in edges:
        union(parent, rank, x, y)
    
    # Group edges by component
    components = defaultdict(list)
    for x, y, z in edges:
        root = find_parent(parent, x)
        components[root].append((x, y, z))
    
    result = [0] * N
    
    # For each component, solve the system
    for root, edge_list in components.items():
        # Collect all nodes in this component
        nodes = set()
        for x, y, _ in edge_list:
            nodes.add(x)
            nodes.add(y)
        nodes = list(nodes)
        node_index = {node: i for i, node in enumerate(nodes)}
        n_nodes = len(nodes)
        
        # Gaussian elimination over GF(2) for each bit
        MAX_BITS = 30
        # For each bit position, we solve a system
        # We'll track the basis and solution for each bit
        
        # But since we want minimal sum, we can do the following:
        # For each component, we can set one variable to 0,
        # and compute others based on that.
        
        # Let's pick the first node in the component as the free variable (set to 0)
        base_node = nodes[0]
        # We'll represent other variables in terms of base_node
        # a[i] = a[base_node] ^ value[i]
        value = {}
        value[base_node] = 0
        
        # Now we need to check consistency and propagate
        # We model this as a graph where each constraint is an edge
        # and we try to assign values consistently
        
        # Build adjacency list with constraints
        adj = defaultdict(list)
        for x, y, z in edge_list:
            adj[x].append((y, z))
            adj[y].append((x, z))
        
        # BFS to assign values
        visited = set()
        queue = [base_node]
        visited.add(base_node)
        value[base_node] = 0
        consistent = True
        
        while queue:
            u = queue.pop()
            for v, xor_val in adj[u]:
                expected = value[u] ^ xor_val
                if v not in visited:
                    visited.add(v)
                    value[v] = expected
                    queue.append(v)
                else:
                    # Check consistency
                    if value[v] != expected:
                        consistent = False
                        break
            if not consistent:
                break
        
        if not consistent:
            print(-1)
            return
        
        # Assign values to result
        for node in nodes:
            if node in value:
                result[node] = value[node]
    
    # Check if all constraints are satisfied
    for x, y, z in edges:
        if result[x] ^ result[y] != z:
            print(-1)
            return
            
    print(' '.join(map(str, result)))

if __name__ == "__main__":
    main()
