import heapq

def minPath(grid, k):
    N = len(grid)
    directions = [(-1,0), (1,0), (0,-1), (0,1)]
    heap = []
    
    for i in range(N):
        for j in range(N):
            heapq.heappush(heap, (grid[i][j], [(i, j)], [(grid[i][j], (i, j))]))
    
    while heap:
        val, path, seq = heapq.heappop(heap)
        if len(seq) == k:
            return [s[0] for s in seq]
        i, j = path[-1]
        for dx, dy in directions:
            ni, nj = i + dx, j + dy
            if 0 <= ni < N and 0 <= nj < N:
                new_val = grid[ni][nj]
                new_path = path + [(ni, nj)]
                new_seq = seq + [(new_val, (ni, nj))]
                heapq.heappush(heap, (new_val, new_path, new_seq))
