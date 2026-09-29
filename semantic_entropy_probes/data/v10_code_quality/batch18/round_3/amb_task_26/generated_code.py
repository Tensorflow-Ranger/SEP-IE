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
            val = A[i] * B[j]
            heapq.heappush(heap, val)
    
    for k in range(N):
        while heap:
            val = heapq.heappop(heap)
            if val == A[i] * B[j]:
                K -= 1
                if K == 0:
                    print(val)
                    return
            else:
                heapq.heappush(heap, val)
                break
    
    print(-1)

main()
