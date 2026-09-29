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
    
    # For each query, perform BFS to find minimum stairs
    for A, B, Y, C, D, Z in queries:
        # Priority queue: (stair_count, row, col, floor)
        pq = [(0, A, B, Y)]
        visited = [[[float('inf')] * (F[i][j] + 1) for j in range(W)] for i in range(H)]
        visited[A][B][Y] = 0
        
        while pq:
            stair_count, r, c, f = heapq.heappop(pq)
            
            if r == C and c == D and f == Z:
                print(stair_count)
                break
            
            # Move up or down within the same building
            for df in [-1, 1]:
                nf = f + df
                if 1 <= nf <= F[r][c]:
                    if visited[r][c][nf] > stair_count + abs(df):
                        visited[r][c][nf] = stair_count + abs(df)
                        heapq.heappush(pq, (visited[r][c][nf], r, c, nf))
            
            # Move to adjacent buildings via skywalkway
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and F[nr][nc] >= f:
                    if visited[nr][nc][f] > stair_count:
                        visited[nr][nc][f] = stair_count
                        heapq.heappush(pq, (visited[nr][nc][f], nr, nc, f))
    
if __name__ == "__main__":
    main()
