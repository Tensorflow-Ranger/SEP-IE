import sys
from itertools import accumulate

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
    B_prefix = [0] + list(accumulate(B))
    W_prefix = [0] + list(accumulate(W))
    
    max_sum = 0  # Choosing no balls gives sum 0
    
    # Try all valid combinations where black_balls >= white_balls
    # Let's say we choose k white balls, then we can choose any number of black balls >= k
    # But to maximize sum, if we choose k white balls, we should choose as many positive-value black balls as possible
    # Actually, let's think differently:
    # For each possible number of white balls chosen (from 0 to M), 
    # determine how many black balls we should choose (at least that many)
    # and compute the max sum for that case
    
    # But more efficient approach:
    # We can iterate through how many white balls we take (say w)
    # Then we need at least w black balls
    # To maximize, we take the top w white balls and top w black balls
    # But we might want to take MORE black balls if they're positive
    
    # So for w white balls taken:
    # Take top w white balls
    # Take at least w black balls, but potentially more if beneficial
    # So we compute sum of top w white balls + sum of top max(w, optimal_number) black balls
    
    # Actually, let's reframe:
    # We want to choose b black balls and w white balls such that b >= w
    # Maximize sum of top b black values + sum of top w white values
    
    # Strategy:
    # For each w from 0 to M:
    #   SumW = sum of top w white balls
    #   We must take at least w black balls
    #   But we can take more black balls if they contribute positively
    #   So take as many black balls as possible whose values are > 0
    #   But we must take at least w of them
    #   So take max(w, number of positive black balls) if we're maximizing
    #   Wait, no. If we've decided to take w white balls, we take exactly w
    #   And we take at least w black balls, but to maximize we might take more
    
    # Let's precompute prefix sums of sorted arrays (in descending order)
    # Prefix sums include 0 at index 0
    
    # For each count of white balls w (0 to M):
    #   Value from white balls = W_prefix[w]
    #   Must take at least w black balls
    #   Can take more if beneficial
    #   Since black balls are sorted descending, 
    #   take as many as possible from the start while value is non-negative,
    #   but at least w
    
    for w in range(M + 1):
        white_sum = W_prefix[w]
        
        # Need at least w black balls
        # But to maximize total, take additional black balls with positive values
        # Black balls are already sorted descending
        # Find how many black balls have positive values
        # But we must take at least w
        
        # In our prefix sum array B_prefix, B_prefix[b] is sum of first b black balls
        # We want to find the best b >= w to maximize B_prefix[b]
        # Since values can be negative, B_prefix is not necessarily increasing
        # But initially it increases while values are positive
        
        # Actually, prefix sum of sorted descending sequence:
        # As long as B[i] >= 0, adding it increases the sum
        # Once B[i] < 0, adding it decreases the sum
        # So the optimal number of black balls to take (to maximize their sum) 
        # is up to the point before negative values decrease the total
        
        # But we must take at least w
        # So we take max(optimal_positive_count, w)
        
        # Find how many initial black balls are non-negative in contribution
        # Actually, this is tricky because we sort descending
        # B[0] >= B[1] >= ... 
        # We want to find largest k such that B[k-1] >= 0
        # Because taking k balls means taking indices 0..k-1
        
        # But actually, to maximize just the black balls sum, 
        # we take all balls with positive value
        # Find the index where values become negative
        # But we must take at least w
        
        pos_black_count = 0
        for val in B:
            if val > 0:
                pos_black_count += 1
            else:
                break
                
        black_balls_to_take = max(w, pos_black_count)
        # But also limited by N
        black_balls_to_take = min(black_balls_to_take, N)
        
        black_sum = B_prefix[black_balls_to_take]
        total = white_sum + black_sum
        if total > max_sum:
            max_sum = total
            
    print(max_sum)

if __name__ == "__main__":
    main()
