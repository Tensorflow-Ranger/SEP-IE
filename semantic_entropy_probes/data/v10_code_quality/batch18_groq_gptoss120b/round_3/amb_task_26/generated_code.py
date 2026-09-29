import sys
from heapq import nlargest, nsmallest
from bisect import bisect_left, bisect_right

def count_less_equal(A, B, C, target):
    """Count how many values A_i*B_j + B_j*C_k + C_k*A_i <= target"""
    n = len(A)
    count = 0
    
    # Sort arrays for efficient counting
    A.sort()
    B.sort()
    C.sort()
    
    # For each B_j, we want to count pairs (i,k) such that:
    # A_i*B_j + B_j*C_k + C_k*A_i <= target
    # Rearranging: A_i*B_j + C_k*A_i + B_j*C_k <= target
    # A_i(B_j + C_k) + B_j*C_k <= target
    
    for j in range(n):
        bj = B[j]
        # For fixed j, we have: A_i*bj + C_k*A_i + bj*C_k <= target
        # A_i(bj + C_k) + bj*C_k <= target
        # A_i(bj + C_k) <= target - bj*C_k
        
        for k in range(n):
            ck = C[k]
            right_side = target - bj * ck
            if right_side < 0 and bj + ck >= 0:
                continue
                
            if bj + ck > 0:
                # A_i <= right_side / (bj + ck)
                threshold = right_side // (bj + ck)
                if right_side % (bj + ck) == 0:
                    threshold -= 1
                # Count A_i <= threshold
                idx = bisect_right(A, threshold)
                count += idx
            elif bj + ck == 0:
                if right_side >= 0:
                    count += n
            else:  # bj + ck < 0
                # A_i >= right_side / (bj + ck) (since we're dividing by negative)
                threshold = right_side // (bj + ck)
                if right_side % (bj + ck) != 0:
                    threshold += 1
                # Count A_i >= threshold
                idx = bisect_left(A, threshold)
                count += n - idx
    
    return count

def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:2+N]))
    B = list(map(int, data[2+N:2+2*N]))
    C = list(map(int, data[2+2*N:2+3*N]))
    
    # We'll use binary search on the answer
    # First, let's find min and max possible values to set our bounds
    
    min_val = float('inf')
    max_val = float('-inf')
    
    # Sample some combinations to get reasonable bounds
    for i in range(min(100, N)):
        for j in range(min(100, N)):
            for k in range(min(100, N)):
                val = A[i] * B[j] + B[j] * C[k] + C[k] * A[i]
                min_val = min(min_val, val)
                max_val = max(max_val, val)
    
    # Expand the bounds generously
    left = min_val - 10**12
    right = max_val + 10**12
    
    def count_le(x):
        # Count number of triples (i,j,k) with A[i]*B[j] + B[j]*C[k] + C[k]*A[i] <= x
        count = 0
        # This is too slow for large N, so we need a better approach
        
        # Let's sort the arrays
        sorted_A = sorted(A)
        sorted_B = sorted(B)
        sorted_C = sorted(C)
        
        # For each B[j], we compute S_j = B[j](A[i] + C[k]) + A[i]*C[k]
        # We can rewrite this as B[j]*A[i] + B[j]*C[k] + A[i]*C[k]
        # = A[i]*(B[j]) + C[k]*(B[j] + A[i])
        
        # Actually, let's think differently:
        # For fixed j,k: we want to count i such that A[i]*B[j] + B[j]*C[k] + C[k]*A[i] <= x
        # A[i]*B[j] + C[k]*A[i] <= x - B[j]*C[k]
        # A[i]*(B[j] + C[k]) <= x - B[j]*C[k]
        
        for j in range(len(sorted_B)):
            bj = sorted_B[j]
            for k in range(len(sorted_C)):
                ck = sorted_C[k]
                rhs = x - bj * ck
                if bj + ck > 0:
                    # A[i] <= rhs / (bj + ck)
                    if rhs < 0:
                        continue
                    threshold = rhs // (bj + ck)
                    if rhs % (bj + ck) == 0 and rhs > 0:
                        threshold -= 1
                    pos = bisect_right(sorted_A, threshold)
                    count += pos
                elif bj + ck == 0:
                    if rhs >= 0:
                        count += len(sorted_A)
                else:  # bj + ck < 0
                    # A[i] >= rhs / (bj + ck) (inequality flips)
                    if rhs <= 0:
                        count += len(sorted_A)
                        continue
                    threshold = rhs // (bj + ck)
                    if rhs % (bj + ck) != 0:
                        threshold += 1
                    pos = bisect_left(sorted_A, threshold)
                    count += len(sorted_A) - pos
                    
                if count >= K + 10**6:  # Early exit optimization
                    return count
                    
        return count
    
    # Binary search for the Kth smallest value
    left, right = -4 * 10**18, 4 * 10**18
    
    while left < right:
        mid = (left + right) // 2
        if count_le(mid) >= K:
            right = mid
        else:
            left = mid + 1
            
    print(left)

solve()
