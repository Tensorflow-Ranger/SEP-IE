import sys

def main():
    # Read input
    line = sys.stdin.readline().strip()
    N, M = map(int, line.split())
    
    # If N == 1, then X = M + 1 (since 1^0 + 1^1 + ... + 1^M = M + 1)
    if N == 1:
        X = M + 1
        print(X)
        return
    
    # For N > 1, we can use the geometric series formula:
    # X = (N^(M+1) - 1) / (N - 1)
    # But we need to be careful about overflow.
    # Since M <= 100 and N <= 10^9, N^M could be very large.
    # However, if the result would be infinite, it means the sum diverges,
    # but since M is finite, the sum is always finite.
    
    # Let's compute the sum directly, but check for potential issues.
    # Actually, since M is at most 100, and we're dealing with integers,
    # the sum might get very large but not infinite.
    
    # The problem says "If X, print the value of X. If X, print inf."
    # This seems like a translation issue. I think it means:
    # Print the value of X if it's computable, otherwise print inf.
    # But mathematically, for finite M, X is always finite.
    
    # Re-reading: It looks like there's missing text. 
    # A common variant is to print "inf" when the sum diverges, 
    # which happens when M approaches infinity and |N| >= 1.
    # But here M is finite, so the sum is always finite.
    
    # Let me assume the condition for printing "inf" is when N=1 and M is infinite,
    # but since M is given and finite, perhaps there's another interpretation.
    
    # Another possibility: if N^M overflows in some specific way or if there's a mod involved.
    # Given the constraints, let's just compute the sum.
    
    # For N != 1, X = (N^(M+1) - 1) / (N - 1)
    # But computing N^(M+1) directly might lead to very large numbers.
    
    # In Python, integers have arbitrary precision, so we can compute this directly.
    
    if M == 0:
        X = 1
        print(X)
        return
        
    # Compute N^(M+1)
    power = pow(N, M + 1)
    # Compute (N^(M+1) - 1) / (N - 1)
    X = (power - 1) // (N - 1)
    print(X)

if __name__ == "__main__":
    main()
