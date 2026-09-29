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
    
    heap = []
    
    for i in range(N):
        for j in range(N):
            val = A[i] * B[j] + B[j] * C[j] + C[j] * A[i]
            heapq.heappush(heap, val)
            if len(heap) > K:
                heapq.heappop(heap)
    
    print(heap[0])

if __name__ == "__main__":
    main()
