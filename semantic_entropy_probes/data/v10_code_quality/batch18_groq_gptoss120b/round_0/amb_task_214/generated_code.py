import sys
sys.setrecursionlimit(1 << 25)

def solve() -> None:
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    edges = []
    for _ in range(M):
        x = int(next(it)) - 1
        y = int(next(it)) - 1
        z = int(next(it))
        edges.append((x, y, z))

    # adjacency list with edge index to retrieve Z quickly
    adj = [[] for _ in range(N)]
    for idx, (x, y, z) in enumerate(edges):
        adj[x].append((y, idx))
        adj[y].append((x, idx))

    # answer array
    ans = [0] * N

    # process each bit separately
    for b in range(31):          # because Z <= 1e9 < 2^30, 31 is safe
        bit_val = [-1] * N       # -1 = unvisited, else 0/1
        for v in range(N):
            if bit_val[v] != -1:
                continue
            # start new component
            stack = [v]
            bit_val[v] = 0
            comp = [v]
            ok = True
            while stack and ok:
                cur = stack.pop()
                cur_val = bit_val[cur]
                for nb, eidx in adj[cur]:
                    c = (edges[eidx][2] >> b) & 1
                    need = cur_val ^ c
                    if bit_val[nb] == -1:
                        bit_val[nb] = need
                        stack.append(nb)
                        comp.append(nb)
                    elif bit_val[nb] != need:
                        ok = False
                        break
            if not ok:
                print(-1)
                return
            # decide whether to flip this component
            ones = sum(bit_val[x] for x in comp)
            if ones > len(comp) - ones:   # flipping gives fewer ones
                for x in comp:
                    bit_val[x] ^= 1
        # add this bit to answer
        for i in range(N):
            if bit_val[i] == 1:
                ans[i] |= (1 << b)

    print(' '.join(map(str, ans)))

if __name__ == "__main__":
    solve()
