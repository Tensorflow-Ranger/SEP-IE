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
        A = int(data[idx]) - 1; idx += 1
        B = int(data[idx]) - 1; idx += 1
        Y = int(data[idx]); idx += 1
        C = int(data[idx]) - 1; idx += 1
        D = int(data[idx]) - 1; idx += 1
        Z = int(data[idx]); idx += 1
        queries.append((A, B, Y, C, D, Z))
    
    INF = float('inf')
    
    def dijkstra(start_row, start_col, start_floor, end_row, end_col, end_floor):
        dist = [[ [INF] * (F[i][j] + 1) for j in range(W)] for i in range(H)]
        dist[start_row][start_col][start_floor] = 0
        heap = [(0, start_row, start_col, start_floor)]
        
        while heap:
            cost, r, c, f = heapq.heappop(heap)
            if r == end_row and c == end_col and f == end_floor:
                return cost
            if cost > dist[r][c][f]:
                continue
            
            # Move up or down stairs
            for df in [-1, 1]:
                nf = f + df
                if 1 <= nf <= F[r][c]:
                    if dist[r][c][nf] > cost + 1:
                        dist[r][c][nf] = cost + 1
                        heapq.heappush(heap, (dist[r][c][nf], r, c, nf))
            
            # Skywalk to adjacent buildings
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and F[nr][nc] >= f:
                    if dist[nr][nc][f] > cost:
                        dist[nr][nc][f] = cost
                        heapq.heappush(heap, (dist[nr][nc][f], nr, nc, f))
        
        return -1
    
    for A, B, Y, C, D, Z in queries:
        res = dijkstra(A, B, Y, C, D, Z)
        print(res)

if __name__ == "__main__":
    main()
