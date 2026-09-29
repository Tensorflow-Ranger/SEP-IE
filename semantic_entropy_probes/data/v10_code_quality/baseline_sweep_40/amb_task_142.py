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
    
    # Precompute for each (i,j,x) the minimum number of stairs needed to reach (i,j,x)
    # We will use BFS for each query, but since Q is large, we need an efficient way
    # Instead, we will use BFS for each query, but optimize by using a priority queue (Dijkstra's algorithm)
    
    def solve_query(a, b, y, c, d, z):
        if a == c and b == d:
            return abs(y - z)
        
        # Priority queue: (stairs_used, current_floor, current_row, current_col)
        pq = []
        heapq.heappush(pq, (0, y, a, b))
        visited = [[dict() for _ in range(W)] for _ in range(H)]
        visited[a][b][y] = 0
        
        while pq:
            stairs, curr_floor, i, j = heapq.heappop(pq)
            
            # Check if we reached the destination
            if i == c and j == d:
                if curr_floor == z:
                    return stairs
                else:
                    continue
            
            # Move up/down stairs
            for delta in [-1, 1]:
                new_floor = curr_floor + delta
                if 1 <= new_floor <= F[i][j]:
                    new_stairs = stairs + 1
                    if new_floor not in visited[i][j] or new_stairs < visited[i][j][new_floor]:
                        visited[i][j][new_floor] = new_stairs
                        heapq.heappush(pq, (new_stairs, new_floor, i, j))
            
            # Move via sky walkway
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                ni, nj = i + dx, j + dy
                if 0 <= ni < H and 0 <= nj < W and F[ni][nj] >= curr_floor:
                    new_stairs = stairs
                    if curr_floor not in visited[ni][nj] or new_stairs < visited[ni][nj].get(curr_floor, float('inf')):
                        visited[ni][nj][curr_floor] = new_stairs
                        heapq.heappush(pq, (new_stairs, curr_floor, ni, nj))
        
        return -1  # Should not happen as per problem constraints
    
    for a, b, y, c, d, z in queries:
        print(solve_query(a, b, y, c, d, z))

if __name__ == "__main__":
    main()
