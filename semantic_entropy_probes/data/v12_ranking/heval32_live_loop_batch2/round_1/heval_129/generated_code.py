import heapq

def minPath(grid, k):
    N = len(grid)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    heap = []
    visited = [[set() for _ in range(N)] for _ in range(N)]
    
    for i in range(N):
        for j in range(N):
            heapq.heappush(heap, (grid[i][j], i, j, [(i, j)]))
    
    while heap:
        val, x, y, path = heapq.heappop(heap)
        if len(path) == k:
            return path
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < N and 0 <= ny < N:
                if (nx, ny) not in visited[x][y]:
                    new_path = path + [(nx, ny)]
                    visited[x][y].add((nx, ny))
                    heapq.heappush(heap, (grid[nx][ny], nx, ny, new_path))
