import sys
from itertools import accumulate

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N, M = int(data[0]), int(data[1])
    B = list(map(int, data[2:2+N]))
    W = list(map(int, data[2+N:2+N+M]))
    
    # Sort in descending order to pick highest values first
    B.sort(reverse=True)
    W.sort(reverse=True)
    
    # Compute prefix sums for efficient range sum queries
    B_prefix = [0] + list(accumulate(B))
    W_prefix = [0] + list(accumulate(W))
    
    max_sum = float('-inf')
    
    # Try all valid combinations where black_count >= white_count
    # We only need to check up to min(N, M) white balls,
    # because if we choose more white balls than black balls, it's invalid
    for w in range(M + 1):
        if w > N:
            break
        # For w white balls, we can choose any number of black balls >= w
        # To maximize, we should choose as many black balls as possible, but
        # there's no benefit in choosing more than w + some optimal amount
        # Actually, let's think differently:
        # For each fixed number of white balls w (0 to M),
        # we can choose any number of black balls b where b >= w and b <= N
        # To maximize the sum, we take the top w white balls (already sorted)
        # and top b black balls. 
        # So for a fixed w, the best is to take top min(N, w + (N-w)) = top N
        # Wait, no. For a fixed w, b can be w, w+1, ..., N
        # The sum is W_prefix[w] + B_prefix[b] where b ranges from w to N
        # Since B_prefix is increasing (we sort B in descending order, 
        # so prefix sum increases), the maximum is when b=N
        # But what if some B values are negative? Then we might not want all N
        # Actually, since we want to maximize and we can choose any subset,
        # we should take all positive values. But we have the constraint on counts.
        
        # Let's reframe:
        # We want to choose b black balls and w white balls with b >= w
        # Maximize sum of values
        # Key insight: among the top k values overall (from combined sorted list),
        # how many are black and how many are white, subject to count constraints
        
        # Even better approach:
        # For each possible number of white balls w (0 to M):
        #   Take top w white balls (W_prefix[w])
        #   We must take at least w black balls: take top w black balls (B_prefix[w])
        #   We can take up to N black balls: take top min(N, w + (N-w)) = top N
        #   But to maximize, among black balls, take all non-negative ones, 
        #   but respect the constraint that we took at least w
        #   So take max(w, number of positive black balls) black balls, 
        #   but not more than N
        
        # Simpler correct approach:
        # For w white balls chosen, we must choose at least w black balls
        # To maximize total value:
        #   - Take top w white balls: W_prefix[w]
        #   - From black balls, take the best set of b balls where b >= w and b <= N
        #   - "Best set" means those with highest values, i.e., first b elements in sorted B
        #   - Value is B_prefix[b]
        #   - To maximize B_prefix[b] subject to w <= b <= N, take b = N
        #   - But what if taking fewer black balls gives a better sum due to negatives?
        #   - If we've taken w white balls, we must take at least w black balls
        #   - Among the black balls, we should take all positive ones, and if that's < w, 
        #     add the least negative ones to reach w
        #   - If positive count >= w, then take exactly those positive ones (or more if beneficial)
        
        # Most precise approach:
        # For a fixed number of white balls w:
        # 1. Take top w white balls: sum = W_prefix[w]
        # 2. Must take at least w black balls: sum += B_prefix[w]
        # 3. Can take up to N black balls. Should we take more?
        #    Check B[w], B[w+1], ..., B[N-1] (in our descending sort)
        #    Take those that are non-negative
        #    But we already included them in B_prefix up to some index
        
        # Clean approach:
        # Precompute for black balls: for each k from 0 to N, 
        # what's the maximum sum we can get by taking exactly k black balls?
        # That's just B_prefix[k] if all top k are good
        # But to handle negatives optimally while respecting minimum count:
        # When forced to take at least w black balls, take exactly the top w if they include all positives
        # or take more if beneficial
        
        # Simpler and correct:
        # For w white balls, we must take b black balls, b >= w
        # To maximize sum, take the b black balls with highest values, b >= w
        # This is B_prefix[b] where b >= w
        # Since B is sorted desc, B_prefix is non-decreasing (assuming non-negative values)
        # But values can be negative, so B_prefix can decrease
        # So for a fixed w, we want max_{b=w..N} B_prefix[b]
        # This is max(B_prefix[w], B_prefix[w+1], ..., B_prefix[N])
        # Which is the maximum of a suffix of B_prefix starting from index w
        
        # Precompute max_B_suffix where max_B_suffix[i] = max(B_prefix[i], ..., B_prefix[N])
        # But actually, let's compute for each w, the max sum we can get
        # which is W_prefix[w] + max_{b=w..N} B_prefix[b]
        
        # So we need max suffix of B_prefix
        max_black_from = [0] * (N + 1)
        max_black_from[N] = B_prefix[N]
        for i in range(N - 1, -1, -1):
            max_black_from[i] = max(B_prefix[i], max_black_from[i + 1])
            
        current_sum = W_prefix[w] + max_black_from[w]
        if current_sum > max_sum:
            max_sum = current_sum
            
    print(max_sum)

if __name__ == "__main__":
    main()
