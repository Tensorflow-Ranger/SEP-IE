import sys
import heapq
from collections import deque

def main():
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    H = int(data[idx])
    idx += 1
    W = int(data[idx])
    idx += 1
    
    F = []
    for _ in range(H):
        row = list(map(int, data[idx:idx+W]))
        idx += W
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
    
    # Directions: up, down, left, right
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # For each query, compute the minimum number of stairs
    for A, B, Y, C, D, Z in queries:
        # BFS with priority queue (Dijkstra's algorithm)
        # State: (row, col, floor, stairs_used)
        # We use a priority queue to always expand the state with the least stairs used
        # We also keep a distance matrix to track the minimum stairs used to reach (row, col, floor)
        dist = [[[float('inf')] * (F[i][j] + 1) for j in range(W)] for i in range(H)]
        pq = []
        heapq.heappush(pq, (0, A, B, Y))
        dist[A][B][Y] = 0
        
        while pq:
            stairs, r, c, f = heapq.heappop(pq)
            if r == C and c == D and f == Z:
                break
            if stairs > dist[r][c][f]:
                continue
            
            # Move up or down within the same building
            for df in [-1, 1]:
                nf = f + df
                if 1 <= nf <= F[r][c]:
                    if dist[r][c][nf] > stairs + abs(df):
                        dist[r][c][nf] = stairs + abs(df)
                        heapq.heappush(pq, (dist[r][c][nf], r, c, nf))
            
            # Move to adjacent buildings using sky walkway
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    if dist[nr][nc][f] > stairs:
                        dist[nr][nc][f] = stairs
                        heapq.heappush(pq, (dist[nr][nc][f], nr, nc, f))
        
        print(dist[C][D][Z])

if __name__ == "__main__":
    main()
