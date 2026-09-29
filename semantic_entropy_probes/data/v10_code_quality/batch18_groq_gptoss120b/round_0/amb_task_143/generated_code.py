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
    # edges
    edges = []
    for i in range(H):
        for j in range(W):
            u = i * W + j
            if i + 1 < H:
                v = (i + 1) * W + j
                w = min(F[u], F[v])
                edges.append((w, u, v))
            if j + 1 < W:
                v = i * W + (j + 1)
                w = min(F[u], F[v])
                edges.append((w, u, v))
    edges.sort(reverse=True)  # descending by weight

    Q = int(next(it))
    queries = []
    for idx in range(Q):
        a = int(next(it)) - 1
        b = int(next(it)) - 1
        y = int(next(it))
        c = int(next(it)) - 1
        d = int(next(it)) - 1
        z = int(next(it))
        u = a * W + b
        v = c * W + d
        queries.append((u, v, y, z, idx))

    # DSU with query lists
    parent = list(range(N))
    size = [1] * N
    qlist = [[] for _ in range(N)]  # list of query ids attached to component
    other = [0] * Q  # other endpoint id for each query
    max_yz = [0] * Q
    for idx, (u, v, y, z, _) in enumerate(queries):
        qlist[u].append(idx)
        qlist[v].append(idx)
        other[idx] = v
        max_yz[idx] = max(y, z)

    M = [0] * Q  # connectivity threshold

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for w, u, v in edges:
        ru = find(u)
        rv = find(v)
        if ru == rv:
            continue
        if size[ru] < size[rv]:
            ru, rv = rv, ru
        # merge rv into ru
        parent[rv] = ru
        size[ru] += size[rv]
        # process queries in smaller list (rv)
        lst_small = qlist[rv]
        lst_big = qlist[ru]
        for qid in lst_small:
            if M[qid]:  # already answered
                continue
            other_root = find(other[qid])
            if other_root == ru:
                M[qid] = w
            else:
                lst_big.append(qid)
        qlist[rv] = []  # clear

    # after processing all edges, any unanswered queries have M = 1 (minimum possible)
    for i in range(Q):
        if M[i] == 0:
            M[i] = 1

    out = [0] * Q
    for (u, v, y, z, idx), m in zip(queries, M):
        diff = abs(y - z)
        extra = max(0, m - max(y, z))
        out[idx] = diff + 2 * extra

    sys.stdout.write("\n".join(map(str, out)))

if __name__ == "__main__":
    solve()
