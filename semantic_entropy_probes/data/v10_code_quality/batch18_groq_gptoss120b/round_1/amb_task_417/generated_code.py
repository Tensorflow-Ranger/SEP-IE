import sys
from collections import deque

def solve() -> None:
    it = iter(sys.stdin.read().split())
    H = int(next(it))
    W = int(next(it))
    grid = [list(next(it).strip()) for _ in range(H)]
    A = int(next(it)) - 1
    B = int(next(it)) - 1
    C = int(next(it)) - 1
    D = int(next(it)) - 1

    INF = 10 ** 9
    dist = [[INF] * W for _ in range(H)]
    dq = deque()
    dist[A][B] = 0
    dq.appendleft((A, B))

    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while dq:
        i, j = dq.popleft()
        if (i, j) == (C, D):
            break
        dcur = dist[i][j]
        for di, dj in dirs:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W:
                w = 0 if grid[ni][nj] == '.' else 1
                nd = dcur + w
                if nd < dist[ni][nj]:
                    dist[ni][nj] = nd
                    if w == 0:
                        dq.appendleft((ni, nj))
                    else:
                        dq.append((ni, nj))

    print(dist[C][D])

if __name__ == "__main__":
    solve()
