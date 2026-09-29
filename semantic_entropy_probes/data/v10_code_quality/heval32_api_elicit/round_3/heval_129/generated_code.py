def minPath(grid, k):
    import heapq
    from collections import deque

    N = len(grid)
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    def get_neighbors(x, y):
        neighbors = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < N and 0 <= ny < N:
                neighbors.append((nx, ny))
        return neighbors

    heap = []
    for i in range(N):
        for j in range(N):
            path = [grid[i][j]]
            heapq.heappush(heap, (path, i, j, set([(i, j)])))

    while heap:
        path, x, y, visited = heapq.heappop(heap)
        if len(path) == k:
            return path
        for nx, ny in get_neighbors(x, y):
            if (nx, ny) not in visited:
                new_path = path + [grid[nx][ny]]
                new_visited = visited.copy()
                new_visited.add((nx, ny))
                heapq.heappush(heap, (new_path, nx, ny, new_visited))
