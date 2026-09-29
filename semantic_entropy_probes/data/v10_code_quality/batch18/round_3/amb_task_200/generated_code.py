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
    # Sort white balls in descending order
    W.sort(reverse=True)
    
    # Use max heap to keep track of the best possible sums
    max_heap = []
    
    # Consider all possible numbers of white balls (from 0 to min(N, M))
    for k in range(0, min(N, M) + 1):
        # Take k white balls and k black balls
        if k <= len(W) and k <= len(B):
            sum_white = sum(W[:k])
            sum_black = sum(B[:k])
            total = sum_white + sum_black
            heapq.heappush(max_heap, -total)
    
    # Also consider taking more black balls than white balls
    for k in range(1, min(N, M) + 1):
        # Take k white balls and k+1 black balls
        if k <= len(W) and k+1 <= len(B):
            sum_white = sum(W[:k])
            sum_black = sum(B[:k+1])
            total = sum_white + sum_black
            heapq.heappush(max_heap, -total)
    
    # The maximum sum is the negative of the smallest element in the max heap
    print(-max_heap[0] if max_heap else 0)

if __name__ == "__main__":
    main()
