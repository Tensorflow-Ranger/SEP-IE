import sys
import heapq
from collections import deque

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
    
    INF = float('inf')
    dist = [[[[INF] * (F[i][j] + 1) for _ in range(W)] for j in range(H)] for i in range(H)]
    
    for i in range(H):
        for j in range(W):
            for f in range(F[i][j] + 1):
                dist[i][j][f] = INF
    
    for i in range(H):
        for j in range(W):
            for f in range(F[i][j] + 1):
                if F[i][j] >= f:
                    dist[i][j][f] = 0
    
    for i in range(H):
        for j in range(W):
            for f in range(F[i][j] + 1):
                if dist[i][j][f] == INF:
                    continue
                # Move up or down
                if f + 1 <= F[i][j]:
                    dist[i][j][f + 1] = min(dist[i][j][f + 1], dist[i][j][f] + 1)
                if f - 1 >= 0:
                    dist[i][j][f - 1] = min(dist[i][j][f - 1], dist[i][j][f] + 1)
                # Move to adjacent buildings
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < H and 0 <= nj < W and F[ni][nj] >= f:
                        dist[ni][nj][f] = min(dist[ni][nj][f], dist[i][j][f])
    
    for A, B, Y, C, D, Z in queries:
        print(dist[C][D][Z])

if __name__ == "__main__":
    main()
