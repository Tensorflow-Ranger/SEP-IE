import sys, bisect, heapq

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

    B.sort()
    # Use a min-heap to generate the K smallest values.
    # For each pair (i, k) we will push the smallest j (index 0) into the heap.
    # Heap element: (value, i, k, j_index)
    heap = []
    for i in range(N):
        Ai = A[i]
        for k in range(N):
            Ck = C[k]
            # compute value for j=0
            val = Ai * B[0] + B[0] * Ck + Ck * Ai
            heap.append((val, i, k, 0))
    heapq.heapify(heap)

    ans = None
    for _ in range(K):
        val, i, k, j_idx = heapq.heappop(heap)
        ans = val
        if j_idx + 1 < N:
            next_j = j_idx + 1
            new_val = A[i] * B[next_j] + B[next_j] * C[k] + C[k] * A[i]
            heapq.heappush(heap, (new_val, i, k, next_j))

    print(ans)

if __name__ == "__main__":
    solve()
