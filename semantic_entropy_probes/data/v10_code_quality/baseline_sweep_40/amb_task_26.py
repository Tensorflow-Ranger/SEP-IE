import sys
import heapq

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:2+N]))
    B = list(map(int, data[2+N:2+2*N]))
    C = list(map(int, data[2+2*N:2+3*N]))
    
    # Precompute all possible values of A_i * B_j + B_j * C_k + C_k * A_i
    # To avoid O(N^3) time, we use a min-heap to keep track of the K smallest values
    
    heap = []
    for i in range(N):
        for j in range(N):
            val = A[i] * B[j] + B[j] * C[0] + C[0] * A[i]
            heapq.heappush(heap, val)
    
    # Now, we need to generate more values to find the K-th smallest
    # We use a set to avoid duplicates and a heap to manage the next candidates
    
    seen = set()
    seen.add((0, 0, 0))
    heap = [(A[0] * B[0] + B[0] * C[0] + C[0] * A[0], 0, 0, 0)]
    
    for _ in range(K):
        val, i, j, k = heapq.heappop(heap)
        if _ == K - 1:
            print(val)
            return
        for di, dj, dk in [(0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 0, -1), (0, -1, 0), (-1, 0, 0)]:
            ni, nj, nk = i + di, j + dj, k + dk
            if 0 <= ni < N and 0 <= nj < N and 0 <= nk < N and (ni, nj, nk) not in seen:
                seen.add((ni, nj, nk))
                heapq.heappush(heap, (A[ni] * B[nj] + B[nj] * C[nk] + C[nk] * A[ni], ni, nj, nk))

if __name__ == "__main__":
    main()
