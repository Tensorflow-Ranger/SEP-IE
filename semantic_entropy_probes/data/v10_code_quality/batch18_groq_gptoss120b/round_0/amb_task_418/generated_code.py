import sys
from collections import deque

def solve() -> None:
    data = sys.stdin.read().splitlines()
    it = iter(data)
    H, W = map(int, next(it).split())
    grid = [list(next(it).strip()) for _ in range(H)]
    A, B, C, D = map(int, next(it).split())
    sx, sy = A - 1, B - 1
    tx, ty = C - 1, D - 1

    INF = 10**9
    dist = [[INF] * W for _ in range(H)]
    dq = deque()
    dist[sx][sy] = 0
    dq.append((sx, sy))

    dirs = [(1,0), (-1,0), (0,1), (0,-1)]

    while dq:
        x, y = dq.popleft()
        d = dist[x][y]
        if (x, y) == (tx, ty):
            break
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if 0 <= nx < H and 0 <= ny < W:
                w = 0 if grid[nx][ny] == '.' else 1
                nd = d + w
                if nd < dist[nx][ny]:
                    dist[nx][ny] = nd
                    if w == 0:
                        dq.appendleft((nx, ny))
                    else:
                        dq.append((nx, ny))

    print(dist[tx][ty])

if __name__ == "__main__":
    solve()
