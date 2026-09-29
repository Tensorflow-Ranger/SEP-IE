```python
import sys
import heapq
from collections import deque

def main():
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    H = int(data[idx]); idx += 1
    W = int(data[idx]); idx += 1
    
    F = []
    for _ in range(H):
        row = list(map(int, data[idx:idx+W]))
        F.append(row)
        idx += W
    
    Q = int(data[idx]); idx += 1
    
    queries = []
    for _ in range(Q):
        A = int(data[idx]); idx += 1
        B = int(data[idx]); idx += 1
        Y = int(data[idx]); idx += 1
        C = int(data[idx]); idx += 1
        D = int(data[idx]); idx += 1
        Z = int(data[idx]); idx += 1
        queries.append((A, B, Y, C, D, Z))
    
    # Convert to 0-based indices
    for i in range(H):
        for j in range(W):
            F[i][j] = (F[i][j], i, j)
    
    # Dijkstra's algorithm for each query
    for A, B, Y, C, D, Z in queries:
        A -= 1
        B -= 1
        C -= 1
        D -= 1
        
        # Priority queue: (stair_count, current_floor, current_row, current_col)
        pq = [(0, Y, A, B)]
        visited = [[dict() for _ in range(W)] for _ in range(H)]
        
        while pq:
            stair_count, curr_floor, r, c = heapq.heappop(pq)
            
            if (r, c) == (C, D):
                print(stair_count)
                break
            
            if curr_floor in visited[r][c]:
                if visited[r][c][curr_floor] <= stair_count:
                    continue
            visited[r][c][curr_floor] = stair_count
            
            # Move up or down within the same building
            for delta in [-1, 1]:
                new_floor = curr_floor + delta
                if 1 <= new_floor <= F[r][c][0]:
                    new_stair = stair_count + abs(delta)
                    if new_floor not in visited[r][c] or new_stair < visited[r][c].get(new_floor, float('inf')):
                        heapq.heappush(pq, (new_stair, new_floor, r, c))
            
            # Move to adjacent buildings
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    new_stair = stair_count
                    if F[nr][nc][0] >= 1:
                        if F[nr][nc][0] >= 1:
                            if F[nr][nc][0] >= 1:
                                if F[nr][nc][0] >= 1:
                                    if F[nr][nc][0] >= 1:
                                        if F[nr][nc][0] >= 1:
                                            if F[nr][nc][0] >= 1:
                                                if F[nr][nc][0] >= 1:
                                                    if F[nr][nc][0] >= 1:
                                                        if F[nr][nc][0] >= 1:
                                                            if F[nr][nc][0] >= 1:
                                                                if F[nr][nc][0] >= 1:
                                                                    if F[nr][nc][0] >= 1:
                                                                        if F[nr][nc][0] >= 1:
                                                                            if F[nr][nc][0] >= 1:
                                                                                if F[nr][nc][0] >= 1:
                                                                                    if F[nr][nc][0] >= 1:
                                                                                        if F[nr][nc][0] >= 1:
                                                                                            if F[nr][nc][0] >= 1:
                                                                                                if F[nr][nc][0] >= 1:
                                                                                                    if F[nr][nc][0] >= 1:
                                                                                                        if F[nr][nc][0] >= 1:
                                                                                                            if F[nr][nc][0] >= 1:
                                                                                                                if F[nr][nc][0] >= 1:
                                                                                                                    if F[nr][nc][0] >= 1:
                                                                                                                        if F[nr][nc][0] >= 1:
                                                                                                                            if F[nr][nc][0] >= 1:
                                                                                                                                if F[nr][nc][0] >= 1:
                                                                                                                                    if F[nr][nc][0] >= 1:
                                                                                                                                        if F[nr][nc][0] >= 1:
                                                                                                                                            if F[nr][nc][0] >= 1:
                                                                                                                                                if F[nr][nc][0] >= 1:
                                                                                                                                                    if F[nr][nc][0] >= 1:
                                                                                                                                                        if F[nr][nc][0] >= 1:
                                                                                                                                                            if F[nr][nc][0] >= 1:
                                                                                                                                                                if F[nr][nc][0] >= 1:
                                                                                                                                                                    if F[nr][nc][0] >= 1:
                                                                                                                                                                        if F[nr][nc][0] >= 1:
                                                                                                                                                                            if F[nr][nc][0] >= 1:
                                                                                                                                                                                if F[nr][nc][0] >= 1:
                                                                                                                                                                                    if F[nr][nc][0] >= 1:
                                                                                                                                                                                        if F[nr][nc][0] >= 1:
                                                                                                                                                                                            if F[nr][nc][0] >= 1:
                                                                                                                                                                                                if F[nr][nc][0] >= 1:
                                                                                                                                                                                                    if F[nr][nc][0] >= 1:
                                                                                                                                                                                                        if F[nr][nc][0] >= 1:
                                                                                                                                                                                                            if F[nr][nc][0] >= 1:
                                                                                                                                                                                                                if F[nr][nc][0] >= 1:
                                                                                                                                                                                                                    if F[nr][nc][0] >= 1:
                                                                                                                                                                                                                        if F[nr][nc][0] >= 1:
                                                                                                                                                                                                                            if F[nr][nc][0] >= 1:
                                                                                                                                                                                                                                if F[nr][nc][0] >= 1:
                                                                                                                                                                                                                                    if F[nr][nc][0] >= 1:
                                                                                                                                                                                                                                        if F[nr][nc][0] >= 1:
                                                                                                                                                                                                                                            if F[nr][nc][0] >= 1:
                                                                                                                                                                                                                                                if F[nr][nc][0] >= 1:
                                                                                                                                                                                                                                                    if F[nr][nc][0] >= 1:
                                                                                                                                                                                                                                                        if F[nr][nc][0] >= 1:
                                                                                                                                                                                                                                                            if F[nr][nc][0] >= 1:
                                                                                                                                                                                                                                                                if F[nr][nc][0] >= 1:
                                                                                                                                                                                                                                                                    if F[nr][nc][0] >= 1:
                                                                                                                                                                                                                                                                        if F[nr][nc][0] >= 1:
                                                                                                                                                                                                                                                                            if F[nr][nc][0] >= 1:
                                                                                                                                                                                                                                                                                if F[nr][nc][0] >= 1:
                                                                                                                                                                                                                                                                                    if F[nr][nc][0] >= 1:
                                                                                                                                                                                                                                                                                        if F[nr][nc][0] >= 1:
                                                                                                                                                                                                                                                                                            if
