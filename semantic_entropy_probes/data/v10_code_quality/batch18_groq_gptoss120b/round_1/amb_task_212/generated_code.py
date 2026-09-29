import sys
from collections import deque

def solve() -> None:
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    M = int(next(it))

    adj = [[] for _ in range(N)]
    for _ in range(M):
        x = int(next(it)) - 1
        y = int(next(it)) - 1
        z = int(next(it))
        adj[x].append((y, z))
        adj[y].append((x, z))

    visited = [False] * N
    offset = [0] * N
    ans = [0] * N

    for start in range(N):
        if visited[start]:
            continue
        # BFS for this component
        q = deque([start])
        visited[start] = True
        offset[start] = 0
        comp = []
        cnt = [0] * 31  # count of offsets with bit k = 1

        while q:
            u = q.popleft()
            comp.append(u)
            val = offset[u]
            for k in range(31):
                if (val >> k) & 1:
                    cnt[k] += 1
            for v, z in adj[u]:
                if not visited[v]:
                    offset[v] = offset[u] ^ z
                    visited[v] = True
                    q.append(v)
                else:
                    if (offset[u] ^ offset[v]) != z:
                        print(-1)
                        return

        sz = len(comp)
        root = 0
        for k in range(31):
            if cnt[k] > sz // 2:
                root |= (1 << k)

        for v in comp:
            ans[v] = root ^ offset[v]

    sys.stdout.write(' '.join(map(str, ans)))

if __name__ == "__main__":
    solve()
