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
    
    # Dijkstra's algorithm for each query
    for A, B, Y, C, D, Z in queries:
        # Priority queue: (stair_count, row, col, floor)
        pq = [(0, A, B, Y)]
        # Visited: (row, col, floor) -> min_stairs
        visited = {}
        
        while pq:
            stairs, r, c, f = heapq.heappop(pq)
            if (r, c, f) in visited:
                if visited[(r, c, f)] <= stairs:
                    continue
            visited[(r, c, f)] = stairs
            
            if r == C and c == D and f == Z:
                print(stairs)
                break
            
            # Move up or down within the same building
            if f < F[r][c]:
                new_f = f + 1
                if (r, c, new_f) not in visited or visited[(r, c, new_f)] > stairs + 1:
                    heapq.heappush(pq, (stairs + 1, r, c, new_f))
            if f > 1:
                new_f = f - 1
                if (r, c, new_f) not in visited or visited[(r, c, new_f)] > stairs + 1:
                    heapq.heappush(pq, (stairs + 1, r, c, new_f))
            
            # Move to adjacent buildings using sky walkway
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and F[nr][nc] >= f:
                    if (nr, nc, f) not in visited or visited[(nr, nc, f)] > stairs:
                        heapq.heappush(pq, (stairs, nr, nc, f))
        else:
            print(-1)

if __name__ == "__main__":
    main()
