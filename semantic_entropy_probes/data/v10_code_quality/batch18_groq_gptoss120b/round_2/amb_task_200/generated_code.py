import sys
import heapq

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    B = list(map(int, data[2:2+N]))
    W = list(map(int, data[2+N:2+N+M]))
    
    # Sort in descending order to pick highest values first
    B.sort(reverse=True)
    W.sort(reverse=True)
    
    # Compute prefix sums for efficiency
    prefix_B = [0] * (N + 1)
    prefix_W = [0] * (M + 1)
    
    for i in range(N):
        prefix_B[i+1] = prefix_B[i] + B[i]
    for j in range(M):
        prefix_W[j+1] = prefix_W[j] + W[j]
    
    max_sum = 0  # Choosing no balls is always an option
    
    # Try all possible numbers of white balls chosen (0 to M)
    # For each, we can choose at least that many black balls
    for w_count in range(min(M, N) + 1):
        # We choose w_count white balls and at least w_count black balls
        # To maximize, we choose exactly w_count black balls (if beneficial)
        # But we can choose more black balls if it increases the sum
        
        white_sum = prefix_W[w_count]
        
        # We can choose from w_count to N black balls
        # We want to find the number of black balls that maximizes the total sum
        # total_sum = white_sum + black_sum
        # We need black_count >= w_count
        
        # The best is to choose up to N black balls, taking all positive ones
        # But we must take at least w_count
        black_sum = 0
        for i in range(min(w_count, N)):
            black_sum += B[i]
        
        # Now we can add more black balls if they are positive
        for i in range(w_count, N):
            if B[i] > 0:
                black_sum += B[i]
            else:
                break
        
        total = white_sum + black_sum
        if total > max_sum:
            max_sum = total
            
    print(max_sum)

if __name__ == "__main__":
    main()
