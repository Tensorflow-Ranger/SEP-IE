def minPath(grid, k):
    import heapq
    from collections import deque

    N = len(grid)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    visited = [[False] * N for _ in range(N)]
    heap = []

    for i in range(N):
        for j in range(N):
            heapq.heappush(heap, (grid[i][j], i, j, [(i, j)]))

    while heap:
        val, x, y, path = heapq.heappop(heap)
        if len(path) == k:
            return path
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < N and 0 <= ny < N and not visited[nx][ny]:
                visited[nx][ny] = True
                new_path = path + [(nx, ny)]
                heapq.heappush(heap, (grid[nx][ny], nx, ny, new_path))
                visited[nx][ny] = False
