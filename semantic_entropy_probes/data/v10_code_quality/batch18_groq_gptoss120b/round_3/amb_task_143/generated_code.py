import heapq
import sys

def dijkstra(graph, start, end):
    pq = [(0, start)]
    dist = {start: 0}
    while pq:
        d, u = heapq.heappop(pq)
        if u == end:
            return d
        if d > dist.get(u, float('inf')):
            continue
        for v, cost in graph[u]:
            alt = d + cost
            if alt < dist.get(v, float('inf')):
                dist[v] = alt
                heapq.heappush(pq, (alt, v))
    return dist.get(end, -1)

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    H = int(data[idx])
    idx += 1
    W = int(data[idx])
    idx += 1
    
    F = []
    for i in range(H):
        row = []
        for j in range(W):
            row.append(int(data[idx]))
            idx += 1
        F.append(row)
    
    Q = int(data[idx])
    idx += 1
    
    queries = []
    for _ in range(Q):
        A = int(data[idx]) - 1
        idx += 1
        B = int(data[idx]) - 1
        idx += 1
        Y = int(data[idx])
        idx += 1
        C = int(data[idx]) - 1
        idx += 1
        D = int(data[idx]) - 1
        idx += 1
        Z = int(data[idx])
        idx += 1
        queries.append((A, B, Y, C, D, Z))
    
    results = []
    
    for A, B, Y, C, D, Z in queries:
        # Build graph for this query
        graph = {}
        # Nodes are (i, j, floor)
        # Stair edges: (i,j,f) -> (i,j,f+1) and (i,j,f-1) with cost 1
        # Walkway edges: (i,j,f) -> (neighbor,f) with cost 0
        
        for i in range(H):
            for j in range(W):
                floors = F[i][j]
                for f in range(1, floors + 1):
                    node = (i, j, f)
                    if node not in graph:
                        graph[node] = []
                    
                    # Stairs
                    if f > 1:
                        neighbor_floor = (i, j, f - 1)
                        if neighbor_floor not in graph:
                            graph[neighbor_floor] = []
                        graph[node].append((neighbor_floor, 1))
                        graph[neighbor_floor].append((node, 1))
                    
                    # Walkways
                    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                        ni, nj = i + dx, j + dy
                        if 0 <= ni < H and 0 <= nj < W:
                            neighbor_node = (ni, nj, f)
                            if neighbor_node not in graph:
                                graph[neighbor_node] = []
                            graph[node].append((neighbor_node, 0))
        
        start_node = (A, B, Y)
        end_node = (C, D, Z)
        
        if start_node not in graph:
            # Create temporary node
            graph[start_node] = []
            # Add stair connections
            floors = F[A][B]
            if Y > 1:
                graph[start_node].append(((A, B, Y - 1), 1))
                if (A, B, Y - 1) in graph:
                    graph[(A, B, Y - 1)].append((start_node, 1))
            if Y < floors:
                graph[start_node].append(((A, B, Y + 1), 1))
                if (A, B, Y + 1) in graph:
                    graph[(A, B, Y + 1)].append((start_node, 1))
            # Add walkway connections
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                ni, nj = A + dx, B + dy
                if 0 <= ni < H and 0 <= nj < W:
                    neighbor_node = (ni, nj, Y)
                    graph[start_node].append((neighbor_node, 0))
                    if neighbor_node in graph:
                        graph[neighbor_node].append((start_node, 0))
                        
        if end_node not in graph:
            graph[end_node] = []
            # Add stair connections
            floors = F[C][D]
            if Z > 1:
                graph[end_node].append(((C, D, Z - 1), 1))
                if (C, D, Z - 1) in graph:
                    graph[(C, D, Z - 1)].append((end_node, 1))
            if Z < floors:
                graph[end_node].append(((C, D, Z + 1), 1))
                if (C, D, Z + 1) in graph:
                    graph[(C, D, Z + 1)].append((end_node, 1))
            # Add walkway connections
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                ni, nj = C + dx, D + dy
                if 0 <= ni < H and 0 <= nj < W:
                    neighbor_node = (ni, nj, Z)
                    if neighbor_node in graph:
                        graph[neighbor_node].append((end_node, 0))
        
        res = dijkstra(graph, start_node, end_node)
        results.append(str(res))
    
    print('\n'.join(results))

if __name__ == "__main__":
    main()
