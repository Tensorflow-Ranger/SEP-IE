import sys
import heapq

def solve() -> None:
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    K = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    B = [int(next(it)) for _ in range(N)]
    C = [int(next(it)) for _ in range(N)]

    # keep only the smallest values because K is at most 5e5
    L = min(N, 1000)          # 1000^2 = 1e6 pairs, enough for K up to 5e5
    A_small = sorted(A)[:L]
    C_small = sorted(C)[:L]
    B_sorted = sorted(B)

    # pre‑compute for each pair (a,c) the base term a*c and the coefficient (a+c)
    pairs = []
    for a in A_small:
        for c in C_small:
            base = a * c
            coeff = a + c
            pairs.append((base, coeff))

    heap = []
    # initialise heap with the first element (using smallest B) for each pair
    for idx, (base, coeff) in enumerate(pairs):
        val = base + coeff * B_sorted[0]
        heap.append((val, idx, 0))   # (value, pair_index, position in B)
    heapq.heapify(heap)

    ans = 0
    for _ in range(K):
        val, pidx, bpos = heapq.heappop(heap)
        ans = val
        if bpos + 1 < len(B_sorted):
            base, coeff = pairs[pidx]
            nval = base + coeff * B_sorted[bpos + 1]
            heapq.heappush(heap, (nval, pidx, bpos + 1))

    print(ans)

if __name__ == "__main__":
    solve()
