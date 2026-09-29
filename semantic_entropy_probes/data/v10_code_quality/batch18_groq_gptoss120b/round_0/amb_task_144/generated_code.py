import sys

sys.setrecursionlimit(1 << 25)
input = sys.stdin.readline

def solve() -> None:
    H, W = map(int, input().split())
    N = H * W
    heights = []
    grid = []
    for i in range(H):
        row = list(map(int, input().split()))
        grid.append(row)
        for j, h in enumerate(row):
            heights.append(h)
    max_h = max(heights)

    Q = int(input())
    queries = []
    for _ in range(Q):
        a, b, y, c, d, z = map(int, input().split())
        a -= 1; b -= 1; c -= 1; d -= 1
        s_idx = a * W + b
        t_idx = c * W + d
        queries.append([s_idx, t_idx, y, z])

    # parallel binary search for maximum floor where start and target are connected
    low = [0] * Q
    high = [max_h] * Q  # feasible range [0, max_h]

    # adjacency offsets
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]

    # precompute cells sorted by height descending
    cells_by_h = {}
    for i in range(H):
        for j in range(W):
            h = grid[i][j]
            cells_by_h.setdefault(h, []).append(i*W + j)

    sorted_heights = sorted(cells_by_h.keys(), reverse=True)

    class DSU:
        __slots__ = ('parent','size')
        def __init__(self, n):
            self.parent = list(range(n))
            self.size = [1]*n
        def find(self, x):
            while self.parent[x]!=x:
                self.parent[x]=self.parent[self.parent[x]]
                x=self.parent[x]
            return x
        def union(self, a, b):
            a=self.find(a); b=self.find(b)
            if a==b: return
            if self.size[a]<self.size[b]:
                a,b=b,a
            self.parent[b]=a
            self.size[a]+=self.size[b]

    while True:
        mids = {}
        any_pending = False
        for idx in range(Q):
            if low[idx] < high[idx]:
                any_pending = True
                mid = (low[idx] + high[idx] + 1)//2
                mids.setdefault(mid, []).append(idx)
        if not any_pending:
            break

        # process mids in descending order, building DSU incrementally
        dsu = DSU(N)
        added = [False]*N
        cur_h_idx = 0  # index in sorted_heights
        # iterate mids descending
        for mid in sorted(mids.keys(), reverse=True):
            # add all cells with height >= mid
            while cur_h_idx < len(sorted_heights) and sorted_heights[cur_h_idx] >= mid:
                h = sorted_heights[cur_h_idx]
                for idx in cells_by_h[h]:
                    added[idx] = True
                    i, j = divmod(idx, W)
                    for di, dj in dirs:
                        ni, nj = i+di, j+dj
                        if 0 <= ni < H and 0 <= nj < W:
                            nidx = ni*W+nj
                            if added[nidx]:
                                dsu.union(idx, nidx)
                cur_h_idx += 1
            # answer queries with this mid
            for qid in mids[mid]:
                s, t, _, _ = queries[qid]
                if added[s] and added[t] and dsu.find(s) == dsu.find(t):
                    low[qid] = mid
                else:
                    high[qid] = mid - 1

    out = []
    for idx, (s, t, y, z) in enumerate(queries):
        f = low[idx]
        ans = abs(y - f) + abs(z - f)
        out.append(str(ans))
    sys.stdout.write("\n".join(out))

if __name__ == "__main__":
    solve()
