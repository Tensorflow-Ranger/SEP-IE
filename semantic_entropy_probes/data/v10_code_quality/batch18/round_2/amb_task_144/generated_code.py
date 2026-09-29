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
    result = []
    
    for A, B, Y, C, D, Z in queries:
        if A == C and B == D:
            result.append(0)
            continue
        
        # Dijkstra's algorithm
        dist = [[INF] * (W * F[i][j] + 1) for i in range(H) for j in range(W)]
        # (row, col, floor, cost)
        heap = []
        heapq.heappush(heap, (0, A, B, Y))
        dist[A][B * F[A][B] + Y] = 0
        
        while heap:
            cost, r, c, f = heapq.heappop(heap)
            if r == C and c == D and f == Z:
                break
            if cost > dist[r][c * F[r][c] + f]:
                continue
            
            # Move up or down within the same building
            for df in [-1, 1]:
                nf = f + df
                if 1 <= nf <= F[r][c]:
                    new_key = r * W * F[r][c] + c * F[r][c] + nf
                    if dist[r][c * F[r][c] + nf] > cost + abs(df):
                        dist[r][c * F[r][c] + nf] = cost + abs(df)
                        heapq.heappush(heap, (dist[r][c * F[r][c] + nf], r, c, nf))
            
            # Move to adjacent buildings using sky walkway
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and F[nr][nc] >= f:
                    new_key = nr * W * F[nr][nc] + nc * F[nr][nc] + f
                    if dist[nr][nc * F[nr][nc] + f] > cost:
                        dist[nr][nc * F[nr][nc] + f] = cost
                        heapq.heappush(heap, (dist[nr][nc * F[nr][nc] + f], nr, nc, f))
        
        result.append(dist[C][D * F[C][D] + Z])
    
    for ans in result:
        print(ans)

if __name__ == "__main__":
    main()
