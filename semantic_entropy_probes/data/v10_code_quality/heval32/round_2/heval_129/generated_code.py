import heapq

def minPath(grid, k):
    N = len(grid)
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    heap = []
    visited = [[False] * N for _ in range(N)]
    
    for i in range(N):
        for j in range(N):
            heapq.heappush(heap, (grid[i][j], i, j, [(grid[i][j])]))
            visited[i][j] = True
    
    while heap:
        val, x, y, path = heapq.heappop(heap)
        if len(path) == k:
            return path
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < N and 0 <= ny < N and not visited[nx][ny]:
                new_path = path + [grid[nx][ny]]
                heapq.heappush(heap, (new_path[-1], nx, ny, new_path))
                visited[nx][ny] = True
