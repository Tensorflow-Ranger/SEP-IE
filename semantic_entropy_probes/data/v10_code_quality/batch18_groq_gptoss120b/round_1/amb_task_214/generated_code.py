import sys

def solve() -> None:
    data = list(map(int, sys.stdin.read().split()))
    if not data:
        return
    it = iter(data)
    N = next(it)
    M = next(it)
    edges = []
    for _ in range(M):
        x = next(it) - 1
        y = next(it) - 1
        z = next(it)
        edges.append((x, y, z))

    max_bit = 0
    for _, _, z in edges:
        if z:
            max_bit = max(max_bit, z.bit_length())
    max_bit = max(max_bit, 1)  # at least one bit to handle zero case

    A = [0] * N

    for b in range(max_bit):
        parent = list(range(N))
        parity = [0] * N  # xor to parent
        size = [1] * N

        def find(v):
            if parent[v] != v:
                r, p = find(parent[v])
                parity[v] ^= p
                parent[v] = r
            return parent[v], parity[v]

        def union(u, v, w):
            ru, pu = find(u)
            rv, pv = find(v)
            if ru == rv:
                return (pu ^ pv) == w  # must satisfy
            # attach smaller to larger
            if size[ru] < size[rv]:
                ru, rv = rv, ru
                pu, pv = pv, pu
            # set parent of rv to ru
            parent[rv] = ru
            # we need parity[rv] such that:
            # pu xor parity_between_roots xor pv = w
            # parity_between_roots = pu ^ pv ^ w
            parity[rv] = pu ^ pv ^ w
            size[ru] += size[rv]
            return True

        ok = True
        for x, y, z in edges:
            w = (z >> b) & 1
            if not union(x, y, w):
                ok = False
                break
        if not ok:
            print(-1)
            return

        # gather component info
        comp_cnt1 = {}
        comp_sz = {}
        for v in range(N):
            r, p = find(v)
            comp_sz[r] = comp_sz.get(r, 0) + 1
            comp_cnt1[r] = comp_cnt1.get(r, 0) + p  # p is value when root=0

        # decide root values
        root_val = {}
        for r in comp_sz:
            cnt0 = comp_cnt1[r]          # ones if root=0
            cnt1 = comp_sz[r] - cnt0     # ones if root=1
            root_val[r] = 0 if cnt0 <= cnt1 else 1

        # assign bits
        for v in range(N):
            r, p = find(v)
            bit = root_val[r] ^ p
            if bit:
                A[v] |= (1 << b)

    print(' '.join(map(str, A)))

if __name__ == "__main__":
    solve()
