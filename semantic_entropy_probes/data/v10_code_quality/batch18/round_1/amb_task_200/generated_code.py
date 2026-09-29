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
    # Sort white balls in ascending order
    W.sort()
    
    # Use a max heap to keep track of the best possible sums
    max_heap = []
    
    # Consider choosing k black balls and k white balls for k from 0 to min(N, M)
    for k in range(0, min(N, M) + 1):
        # Take first k black balls
        sum_black = sum(B[:k])
        # Take last k white balls (smallest k white balls)
        sum_white = sum(W[:k])
        total = sum_black - sum_white
        heapq.heappush(max_heap, -total)
    
    # The maximum sum is the negative of the smallest element in the max heap
    print(-max_heap[0])

if __name__ == "__main__":
    main()
