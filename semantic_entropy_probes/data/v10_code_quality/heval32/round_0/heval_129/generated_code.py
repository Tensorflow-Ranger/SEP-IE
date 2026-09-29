import heapq

def minPath(grid, k):
    import heapq
    N = len(grid)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    min_heap = []
    visited = set()
    
    for i in range(N):
        for j in range(N):
            heapq.heappush(min_heap, (grid[i][j], i, j, [(i, j)]))
    
    while min_heap:
        val, x, y, path = heapq.heappop(min_heap)
        if len(path) == k:
            return path
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < N and 0 <= ny < N and (nx, ny) not in visited:
                visited.add((nx, ny))
                new_path = path + [(nx, ny)]
                heapq.heappush(min_heap, (grid[nx][ny], nx, ny, new_path))
