import sys
from collections import deque
import heapq

def dijkstra(graph, start):
    dist = {start: 0}
    pq = [(0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, float('inf')):
            continue
        for v, cost in graph[u]:
            alt = d + cost
            if alt < dist.get(v, float('inf')):
                dist[v] = alt
                heapq.heappush(pq, (alt, v))
    return dist

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    H = int(data[idx]); idx += 1
    W = int(data[idx]); idx += 1
    
    F = [[0]*W for _ in range(H)]
    for i in range(H):
        for j in range(W):
            F[i][j] = int(data[idx])
            idx += 1
            
    Q = int(data[idx]); idx += 1
    
    queries = []
    for _ in range(Q):
        A = int(data[idx]) - 1; idx += 1
        B = int(data[idx]) - 1; idx += 1
        Y = int(data[idx]); idx += 1
        C = int(data[idx]) - 1; idx += 1
        D = int(data[idx]) - 1; idx += 1
        Z = int(data[idx]); idx += 1
        queries.append((A, B, Y, C, D, Z))
        
    # Precompute for each floor level the reachable blocks
    max_floors = max(max(row) for row in F)
    
    # For each floor level f (1-based), build a graph of connected blocks
    # But we'll do Dijkstra per query instead for accuracy
    # Let's build a global graph where nodes are (i,j,floor) and edges represent movements
    
    # However, that might be too large. Instead, for each query, we run Dijkstra
    # State in Dijkstra: (row, col, floor)
    # Transitions:
    #   Stairs: (i,j,f) -> (i,j,f+1) or (i,j,f-1) with cost |f1-f2|
    #   Walkway: (i,j,f) -> (i',j',f) if both buildings have >= f floors, cost 0
    
    def get_neighbors(i, j):
        neighbors = []
        for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W:
                neighbors.append((ni, nj))
        return neighbors
    
    results = []
    for A, B, Y, C, D, Z in queries:
        # Run Dijkstra from (A, B, Y) to (C, D, Z)
        # State: (i, j, floor)
        start = (A, B, Y)
        end = (C, D, Z)
        
        # To avoid recomputing, we can BFS/Dijkstra
        # Distance dictionary
        dist = {}
        dist[start] = 0
        # Priority queue: (distance, i, j, floor)
        pq = [(0, A, B, Y)]
        heapq.heapify(pq)
        
        while pq:
            d, i, j, f = heapq.heappop(pq)
            if d > dist.get((i,j,f), float('inf')):
                continue
            if (i,j,f) == end:
                results.append(str(d))
                break
                
            # Stairs transitions: move up/down within same building
            if f > 1:
                # Move down to floor f-1
                nf = f - 1
                nd = d + 1
                key = (i, j, nf)
                if nd < dist.get(key, float('inf')):
                    dist[key] = nd
                    heapq.heappush(pq, (nd, i, j, nf))
            if f < F[i][j]:
                # Move up to floor f+1
                nf = f + 1
                nd = d + 1
                key = (i, j, nf)
                if nd < dist.get(key, float('inf')):
                    dist[key] = nd
                    heapq.heappush(pq, (nd, i, j, nf))
                    
            # Walkway transitions: move to adjacent block at same floor
            for ni, nj in get_neighbors(i, j):
                if F[ni][nj] >= f:
                    key = (ni, nj, f)
                    if d < dist.get(key, float('inf')):
                        dist[key] = d
                        heapq.heappush(pq, (d, ni, nj, f))
        else:
            # In case end was never reached (shouldn't happen due to constraints)
            results.append("0")
                        
    print('\n'.join(results))

if __name__ == "__main__":
    main()
