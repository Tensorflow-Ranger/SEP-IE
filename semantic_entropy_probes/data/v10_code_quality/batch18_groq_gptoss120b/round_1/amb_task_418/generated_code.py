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
        i, j = dq.popleft()
        d = dist[i][j]
        if (i, j) == (C, D):
            print(d)
            return
        for di, dj in dirs:
            for step in (1, 2):
                ni = i + di * step
                nj = j + dj * step
                if not (0 <= ni < H and 0 <= nj < W):
                    continue
                nd = d + (0 if grid[ni][nj] == '.' else 1)
                if nd < dist[ni][nj]:
                    dist[ni][nj] = nd
                    if grid[ni][nj] == '.':
                        dq.appendleft((ni, nj))
                    else:
                        dq.append((ni, nj))
                    # after paying a kick, the cell becomes road for future moves
                    grid[ni][nj] = '.'

    # guaranteed reachable
    print(dist[C][D])

if __name__ == "__main__":
    solve()
