import heapq
import sys

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    H = int(data[idx])
    idx += 1
    W = int(data[idx])
    idx += 1
    
    # Read floor counts
    F = []
    for i in range(H):
        row = []
        for j in range(W):
            row.append(int(data[idx]))
            idx += 1
        F.append(row)
    
    Q = int(data[idx])
    idx += 1
    
    # Precompute for each cell (i,j): min_cost_to_floor[X] = minimum stairs to reach floor X in building (i,j)
    # But actually, we'll do 0-indexed internally. Let's shift everything to 0-indexed for convenience.
    
    # For each (i,j), we can precompute the minimal stairs needed to move between floors.
    # But since we can move between buildings on the same floor via walkways (cost 0),
    # this becomes a shortest path problem on a graph where:
    # - Nodes are (i, j, floor)
    # - Edges:
    #   - Stairs: (i,j,f) -> (i,j,f+1) or (i,j,f-1) with cost 1
    #   - Walkways: (i,j,f) -> (i',j',f) with cost 0 if adjacent
    
    # However, doing this for all floors for all cells is too much.
    # Instead, for each query, we run Dijkstra on the graph with state (row, col, floor).
    
    # Let's parse queries
    queries = []
    for _ in range(Q):
        A = int(data[idx]) - 1  # to 0-indexed
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
        # Dijkstra from (A, B, Y) to any (C, D, Z)
        # State: (cost, row, col, floor)
        heap = [(0, A, B, Y)]
        # dist[row][col][floor] = min cost to reach (row, col, floor)
        # To save space, use a dictionary
        dist = {}
        dist[(A, B, Y)] = 0
        
        while heap:
            cost, i, j, f = heapq.heappop(heap)
            if cost > dist.get((i, j, f), float('inf')):
                continue
            if i == C and j == D and f == Z:
                results.append(cost)
                break
            # Move up/down stairs
            if f > 1:
                nf = f - 1
                nc = cost + 1
                if nc < dist.get((i, j, nf), float('inf')):
                    dist[(i, j, nf)] = nc
                    heapq.heappush(heap, (nc, i, j, nf))
            if f < F[i][j]:
                nf = f + 1
                nc = cost + 1
                if nc < dist.get((i, j, nf), float('inf')):
                    dist[(i, j, nf)] = nc
                    heapq.heappush(heap, (nc, i, j, nf))
            # Move via walkways to adjacent cells on same floor
            for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W:
                    nf = f
                    nc = cost
                    if nc < dist.get((ni, nj, nf), float('inf')):
                        dist[(ni, nj, nf)] = nc
                        heapq.heappush(heap, (nc, ni, nj, nf))
        else:
            # Should not happen as per problem statement
            results.append(-1)
    
    # Output results
    for res in results:
        print(res)

if __name__ == "__main__":
    main()
