def minPath(grid, k):
    import heapq
    from collections import deque

    n = len(grid)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def get_neighbors(x, y):
        neighbors = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < n:
                neighbors.append((nx, ny))
        return neighbors

    min_heap = []
    visited = set()

    for i in range(n):
        for j in range(n):
            val = grid[i][j]
            path = [val]
            heapq.heappush(min_heap, (val, path, i, j, set([(i, j)])))

    while min_heap:
        val, path, x, y, visited_cells = heapq.heappop(min_heap)
        if len(path) == k:
            return path
        for nx, ny in get_neighbors(x, y):
            if (nx, ny) not in visited_cells:
                new_path = path + [grid[nx][ny]]
                new_visited = visited_cells.copy()
                new_visited.add((nx, ny))
                heapq.heappush(min_heap, (new_path[-1], new_path, nx, ny, new_visited))

    return []
