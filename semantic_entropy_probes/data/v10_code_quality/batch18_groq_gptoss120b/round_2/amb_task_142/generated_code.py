import sys
from collections import deque
import heapq

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    H = int(data[idx])
    W = int(data[idx+1])
    idx += 2
    
    F = []
    for i in range(H):
        row = []
        for j in range(W):
            row.append(int(data[idx]))
            idx += 1
        F.append(row)
    
    Q = int(data[idx])
    idx += 1
    
    queries = []
    for _ in range(Q):
        A = int(data[idx]) - 1
        B = int(data[idx+1]) - 1
        Y = int(data[idx+2])
        C = int(data[idx+3]) - 1
        D = int(data[idx+4]) - 1
        Z = int(data[idx+5])
        idx += 6
        queries.append((A, B, Y, C, D, Z))
    
    # Precompute for each floor height, which cells have that height or more
    max_floors = max(max(row) for row in F)
    floors_cells = [[] for _ in range(max_floors + 1)]
    for i in range(H):
        for j in range(W):
            f = F[i][j]
            for k in range(1, f+1):
                floors_cells[k].append((i,j))
    
    # For each floor level, build a graph of connected components (walkways allowed)
    # We'll do BFS/DFS to find connected components for each floor level
    comp_id = [[[-1]*W for _ in range(H)] for _ in range(max_floors + 1)]
    components = [[] for _ in range(max_floors + 1)]
    
    for floor in range(1, max_floors + 1):
        cid = 0
        visited = [[False]*W for _ in range(H)]
        for i in range(H):
            for j in range(W):
                if F[i][j] >= floor and not visited[i][j]:
                    # BFS to find component
                    queue = [(i,j)]
                    current_comp = []
                    visited[i][j] = True
                    comp_id[floor][i][j] = cid
                    current_comp.append((i,j))
                    index = 0
                    while index < len(queue):
                        x, y = queue[index]
                        index += 1
                        # Neighbors
                        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nx, ny = x+dx, y+dy
                            if 0 <= nx < H and 0 <= ny < W:
                                if not visited[nx][ny] and F[nx][ny] >= floor:
                                    visited[nx][ny] = True
                                    comp_id[floor][nx][ny] = cid
                                    current_comp.append((nx,ny))
                                    queue.append((nx,ny))
                    components[floor].append(current_comp)
                    cid += 1
    
    # Now for each query, we do Dijkstra on states (row, col, floor)
    results = []
    for A, B, Y, C, D, Z in queries:
        if (A,B,Y) == (C,D,Z):
            results.append('0')
            continue
            
        # State: (cost, row, col, floor)
        heap = [(0, A, B, Y)]
        # dist[row][col][floor] = min cost to reach (row, col, floor)
        dist = {}
        dist[(A,B,Y)] = 0
        
        found = False
        res = 0
        
        while heap:
            cost, i, j, f = heapq.heappop(heap)
            if cost > dist.get((i,j,f), float('inf')):
                continue
            if (i,j,f) == (C,D,Z):
                results.append(str(cost))
                found = True
                break
                
            # Stairs: move up/down within same building
            if f > 1:
                nf = f - 1
                if (i,j,nf) not in dist or cost + 1 < dist[(i,j,nf)]:
                    dist[(i,j,nf)] = cost + 1
                    heapq.heappush(heap, (cost + 1, i, j, nf))
            if f < F[i][j]:
                nf = f + 1
                if (i,j,nf) not in dist or cost + 1 < dist[(i,j,nf)]:
                    dist[(i,j,nf)] = cost + 1
                    heapq.heappush(heap, (cost + 1, i, j, nf))
                    
            # Walkways: move to adjacent building with at least f floors
            for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W:
                    if F[ni][nj] >= f:
                        if (ni,nj,f) not in dist or cost < dist[(ni,nj,f)]:
                            dist[(ni,nj,f)] = cost
                            heapq.heappush(heap, (cost, ni, nj, f))
        
        if not found:
            results.append('-1')  # Should not happen per problem statement
            
    print('\n'.join(results))

if __name__ == "__main__":
    main()
