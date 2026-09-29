import sys
from collections import deque

def solve() -> None:
    it = iter(sys.stdin.read().split())
    H = int(next(it)); W = int(next(it))
    grid = [list(next(it).strip()) for _ in range(H)]
    A = int(next(it)) - 1
    B = int(next(it)) - 1
    C = int(next(it)) - 1
    D = int(next(it)) - 1

    INF = 10**9
    dist = [[INF] * W for _ in range(H)]
    dq = deque()
    dist[A][B] = 0
    dq.append((A, B))

    dirs = [(-1,0),(1,0),(0,-1),(0,1)]

    while dq:
        x, y = dq.popleft()
        d = dist[x][y]
        if (x, y) == (C, D):
            # early exit possible
            continue
        for dx, dy in dirs:
            for step in (1, 2):
                nx = x + dx * step
                ny = y + dy * step
                if not (0 <= nx < H and 0 <= ny < W):
                    break
                nd = d + (1 if grid[nx][ny] == '#' else 0)
                if nd < dist[nx][ny]:
                    dist[nx][ny] = nd
                    if grid[nx][ny] == '#':
                        dq.append((nx, ny))          # cost 1 -> right side
                    else:
                        dq.appendleft((nx, ny))      # cost 0 -> left side
                # if it's a wall we turned it into road after the kick,
                # future expansions will treat it as road automatically
                # (no need to modify grid because cost already accounts)
    print(dist[C][D])

if __name__ == "__main__":
    solve()
