import sys
import heapq

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    B = list(map(int, data[2:2+N]))
    W = list(map(int, data[2+N:2+N+M]))
    
    # Sort black balls in descending order
    B.sort(reverse=True)
    # Sort white balls in ascending order (to take smallest first)
    W.sort()
    
    # Use max heap to keep track of the best possible sum
    max_heap = []
    current_sum = 0
    count = 0
    
    # Try taking k black balls and k white balls for k from 0 to min(N, M)
    for k in range(0, min(N, M) + 1):
        # Take k black balls
        for i in range(k):
            current_sum += B[i]
        # Take k white balls (smallest k)
        for i in range(k):
            current_sum += W[i]
        # Push current_sum to max heap
        heapq.heappush(max_heap, -current_sum)
        # Reset current_sum for next k
        current_sum = 0
    
    # The maximum sum is the negative of the smallest element in the max heap
    print(-max_heap[0])

if __name__ == "__main__":
    main()
