import heapq

def minPath(grid, k):
    N = len(grid)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    heap = []
    
    for i in range(N):
        for j in range(N):
            heapq.heappush(heap, (grid[i][j], i, j, [(grid[i][j], i, j)]))
    
    while k > 1:
        val, x, y, path = heapq.heappop(heap)
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < N and 0 <= ny < N:
                new_val = grid[nx][ny]
                new_path = path + [(new_val, nx, ny)]
                heapq.heappush(heap, (new_val, nx, ny, new_path))
        k -= 1
    
    return [val for val, _, _ in heapq.heappop(heap)[3]]
