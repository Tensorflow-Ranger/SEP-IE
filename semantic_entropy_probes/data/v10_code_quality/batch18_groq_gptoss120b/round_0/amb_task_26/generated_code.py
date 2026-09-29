import sys
import heapq

def solve() -> None:
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    it = iter(input_data)
    N = int(next(it))
    K = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    B = [int(next(it)) for _ in range(N)]
    C = [int(next(it)) for _ in range(N)]

    A.sort()
    B.sort()
    C.sort()

    # limit size of considered A and C to keep heap manageable
    L = min(N, int(K ** 0.5) + 3)
    A = A[:L]
    C = C[:L]

    heap = []
    for i, a in enumerate(A):
        for k, c in enumerate(C):
            base = a * c
            coeff = a + c
            # first j = 0
            val = coeff * B[0] + base
            heap.append((val, i, k, 0, coeff, base))
    heapq.heapify(heap)

    ans = None
    for _ in range(K):
        val, i, k, j_idx, coeff, base = heapq.heappop(heap)
        ans = val
        if j_idx + 1 < N:
            nj = j_idx + 1
            nval = coeff * B[nj] + base
            heapq.heappush(heap, (nval, i, k, nj, coeff, base))
    print(ans)

if __name__ == "__main__":
    solve()
