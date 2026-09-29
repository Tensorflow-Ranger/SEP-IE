import heapq

def minPath(grid, k):
    N = len(grid)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    visited = [[False] * N for _ in range(N)]
    heap = [(grid[0][0], 0, 0, [grid[0][0]])]
    visited[0][0] = True
    
    while heap:
        val, row, col, path = heapq.heappop(heap)
        if len(path) == k:
            return path
        for dr, dc in directions:
            nr, nc = row + dr, col + dc
            if 0 <= nr < N and 0 <= nc < N and not visited[nr][nc]:
                visited[nr][nc] = True
                new_path = path + [grid[nr][nc]]
                heapq.heappush(heap, (grid[nr][nc], nr, nc, new_path))
