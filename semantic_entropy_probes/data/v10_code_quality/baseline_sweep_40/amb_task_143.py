import sys
import heapq
from collections import deque

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    H = int(data[idx]); idx += 1
    W = int(data[idx]); idx += 1
    
    F = []
    for _ in range(H):
        row = list(map(int, data[idx:idx+W]))
        idx += W
        F.append(row)
    
    Q = int(data[idx]); idx += 1
    
    queries = []
    for _ in range(Q):
        A = int(data[idx]); idx += 1
        B = int(data[idx]); idx += 1
        Y = int(data[idx]); idx += 1
        C = int(data[idx]); idx += 1
        D = int(data[idx]); idx += 1
        Z = int(data[idx]); idx += 1
        queries.append((A-1, B-1, Y, C-1, D-1, Z))
    
    # Precompute for each (i,j,f) the minimum number of stairs needed to reach (i,j,f)
    # We will use BFS for each query, but since Q is large, we need an efficient way
    # Instead, we will use BFS for each query, but optimize by using a priority queue (Dijkstra's algorithm)
    
    # For each query, we perform a Dijkstra's algorithm on the graph where nodes are (i, j, f)
    # and edges are:
    # - (i, j, f) -> (i, j, f+1) with cost 1 (stairs up)
    # - (i, j, f) -> (i, j, f-1) with cost 1 (stairs down)
    # - (i, j, f) -> (i', j', f) with cost 0 (sky walkway to adjacent block)
    
    # We will use a priority queue to find the minimum number of stairs used
    
    results = []
    
    for A, B, Y, C, D, Z in queries:
        # Dijkstra's algorithm
        dist = [[[-1] * (F[i][j] + 1) for j in range(W)] for i in range(H)]
        heap = []
        heapq.heappush(heap, (0, A, B, Y))
        dist[A][B][Y] = 0
        
        while heap:
            cost, i, j, f = heapq.heappop(heap)
            if i == C and j == D and f == Z:
                break
            if dist[i][j][f] < cost:
                continue
            # Stairs up
            if f + 1 <= F[i][j]:
                if dist[i][j][f + 1] == -1 or dist[i][j][f + 1] > cost + 1:
                    dist[i][j][f + 1] = cost + 1
                    heapq.heappush(heap, (cost + 1, i, j, f + 1))
            # Stairs down
            if f - 1 >= 0:
                if dist[i][j][f - 1] == -1 or dist[i][j][f - 1] > cost + 1:
                    dist[i][j][f - 1] = cost + 1
                    heapq.heappush(heap, (cost + 1, i, j, f - 1))
            # Sky walkways to adjacent blocks
            for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W:
                    if dist[ni][nj][f] == -1 or dist[ni][nj][f] > cost:
                        dist[ni][nj][f] = cost
                        heapq.heappush(heap, (cost, ni, nj, f))
        
        results.append(str(dist[C][D][Z]))
    
    print('\n'.join(results))

if __name__ == "__main__":
    main()
