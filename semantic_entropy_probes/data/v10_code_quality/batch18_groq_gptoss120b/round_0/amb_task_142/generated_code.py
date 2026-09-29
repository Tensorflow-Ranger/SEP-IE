import sys

sys.setrecursionlimit(1 << 25)

def solve() -> None:
    it = iter(sys.stdin.read().split())
    H = int(next(it)); W = int(next(it))
    N = H * W
    F = [0] * N
    for i in range(H):
        for j in range(W):
            F[i * W + j] = int(next(it))
    Q = int(next(it))
    queries = []
    cell_queries = [[] for _ in range(N)]
    for qid in range(Q):
        a = int(next(it)) - 1
        b = int(next(it)) - 1
        y = int(next(it))
        c = int(next(it)) - 1
        d = int(next(it)) - 1
        z = int(next(it))
        u = a * W + b
        v = c * W + d
        queries.append([u, v, y, z, 0])   # last field will store M
        cell_queries[u].append(qid)
        cell_queries[v].append(qid)

    # DSU
    parent = list(range(N))
    size = [1] * N
    active = [False] * N

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]

    # cells sorted by height descending
    cells = sorted(range(N), key=lambda idx: F[idx], reverse=True)

    for idx in cells:
        h = F[idx]
        active[idx] = True
        i, j = divmod(idx, W)
        # union with active neighbours
        if i > 0 and active[idx - W]:
            union(idx, idx - W)
        if i + 1 < H and active[idx + W]:
            union(idx, idx + W)
        if j > 0 and active[idx - 1]:
            union(idx, idx - 1)
        if j + 1 < W and active[idx + 1]:
            union(idx, idx + 1)

        # check queries attached to this cell
        for qid in cell_queries[idx]:
            if queries[qid][4]:  # already answered
                continue
            u, v = queries[qid][0], queries[qid][1]
            if active[u] and active[v] and find(u) == find(v):
                queries[qid][4] = h  # M found

    out_lines = []
    for u, v, y, z, M in queries:
        # M is guaranteed to be set (grid is fully connected eventually)
        if M >= min(y, z) and M <= max(y, z):
            ans = abs(y - z)
        elif M < min(y, z):
            ans = (y - M) + (z - M)
        else:  # M > max(y, z)
            mx = max(y, z)
            ans = (mx - y) + (mx - z)
        out_lines.append(str(ans))
    sys.stdout.write("\n".join(out_lines))

if __name__ == "__main__":
    solve()
