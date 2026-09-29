def minPath(grid, k):
    import heapq
    from collections import deque

    N = len(grid)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

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
            heapq.heappush(heap, (grid[i][j], i, j, [(i, j)]))

    while k > 1:
        val, x, y, path = heapq.heappop(heap)
        for nx, ny in get_neighbors(x, y):
            if (nx, ny) not in path:
                new_path = path + [(nx, ny)]
                heapq.heappush(heap, (grid[nx][ny], nx, ny, new_path))
        k -= 1

    return [grid[i][j] for i, j in heapq.heappop(heap)[3]]
