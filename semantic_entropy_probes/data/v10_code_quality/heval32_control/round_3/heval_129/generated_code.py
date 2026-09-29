import heapq

def minPath(grid, k):
    N = len(grid)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    heap = []
    visited = set()

    for i in range(N):
        for j in range(N):
            val = grid[i][j]
            heapq.heappush(heap, (val, i, j, [(i, j)]))
            visited.add((i, j))

    while heap:
        val, x, y, path = heapq.heappop(heap)
        if len(path) == k:
            return path
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < N and 0 <= ny < N and (nx, ny) not in visited:
                new_path = path + [(nx, ny)]
                heapq.heappush(heap, (grid[nx][ny], nx, ny, new_path))
                visited.add((nx, ny))
