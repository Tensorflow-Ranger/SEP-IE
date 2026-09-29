import sys
from collections import deque

def solve() -> None:
    it = iter(sys.stdin.read().split())
    H = int(next(it)); W = int(next(it))
    S = [list(next(it).strip()) for _ in range(H)]
    A = int(next(it)) - 1
    B = int(next(it)) - 1
    C = int(next(it)) - 1
    D = int(next(it)) - 1

    INF = 10**9
    dist = [[INF]*W for _ in range(H)]
    dq = deque()
    dist[A][B] = 0
    dq.append((A, B))

    dirs = [(-1,0),(1,0),(0,-1),(0,1)]

    while dq:
        x, y = dq.popleft()
        d = dist[x][y]
        if (x, y) == (C, D):
            print(d)
            return
        # cost 0 moves
        for dx, dy in dirs:
            nx, ny = x+dx, y+dy
            if 0 <= nx < H and 0 <= ny < W and S[nx][ny] == '.' and dist[nx][ny] > d:
                dist[nx][ny] = d
                dq.appendleft((nx, ny))
        # cost 1 moves (kick)
        nd = d + 1
        for dx, dy in dirs:
            # one step
            nx, ny = x+dx, y+dy
            if 0 <= nx < H and 0 <= ny < W and dist[nx][ny] > nd:
                dist[nx][ny] = nd
                dq.append((nx, ny))
            # two steps
            nx2, ny2 = x+2*dx, y+2*dy
            if 0 <= nx2 < H and 0 <= ny2 < W and dist[nx2][ny2] > nd:
                dist[nx2][ny2] = nd
                dq.append((nx2, ny2))
    # should always be reachable
    print(dist[C][D])

if __name__ == "__main__":
    solve()
