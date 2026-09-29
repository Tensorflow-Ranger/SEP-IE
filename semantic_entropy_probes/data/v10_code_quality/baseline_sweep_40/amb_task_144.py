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
    
    # Precompute for each (i,j) and floor x, the minimum stairs needed to reach (i,j,x)
    # We'll use BFS with Dijkstra's algorithm for each query
    # Since Q is large, we can't precompute all possible (i,j,x)
    # So we'll handle each query separately with Dijkstra's algorithm
    
    def solve_query(a, b, y, c, d, z):
        # Priority queue: (stairs_used, current_floor, current_i, current_j)
        pq = [(0, y, a, b)]
        visited = [[dict() for _ in range(W)] for _ in range(H)]
        visited[a][b][y] = 0
        
        while pq:
            stairs, curr_floor, i, j = heapq.heappop(pq)
            if i == c and j == d and curr_floor == z:
                return stairs
            # Move up/down within the same building
            for delta in [-1, 1]:
                new_floor = curr_floor + delta
                if 1 <= new_floor <= F[i][j]:
                    if new_floor not in visited[i][j] or stairs + abs(delta) < visited[i][j][new_floor]:
                        visited[i][j][new_floor] = stairs + abs(delta)
                        heapq.heappush(pq, (visited[i][j][new_floor], new_floor, i, j))
            # Move to adjacent buildings with at least curr_floor floors
            for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W and F[ni][nj] >= curr_floor:
                    if curr_floor not in visited[ni][nj] or stairs < visited[ni][nj][curr_floor]:
                        visited[ni][nj][curr_floor] = stairs
                        heapq.heappush(pq, (visited[ni][nj][curr_floor], curr_floor, ni, nj))
        return -1  # Should not happen as per problem constraints
    
    for a, b, y, c, d, z in queries:
        print(solve_query(a, b, y, c, d, z))

if __name__ == "__main__":
    main()
