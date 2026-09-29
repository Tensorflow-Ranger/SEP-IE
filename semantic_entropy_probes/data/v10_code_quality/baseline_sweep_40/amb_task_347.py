import sys
input = sys.stdin.read

def solve():
    data = input().split()
    idx = 0
    T = int(data[idx])
    idx += 1
    results = []
    
    for _ in range(T):
        N = int(data[idx])
        idx += 1
        A = data[idx]
        idx += 1
        B = data[idx]
        idx += 1
        
        # Preprocess positions of 1s in A
        positions_A = [i for i in range(N) if A[i] == '1']
        # Preprocess positions of 1s in B
        positions_B = [i for i in range(N) if B[i] == '1']
        
        # Check if the number of 1s in A and B are the same
        if len(positions_A) != len(positions_B):
            results.append("-1")
            continue
        
        # Sort positions for both A and B
        positions_A.sort()
        positions_B.sort()
        
        # Check if it's possible to reach B from A
        # For each piece in A, it must be moved to a position in B
        # We need to match each piece in A to a position in B
        # The minimal number of operations is the sum of absolute differences
        # between corresponding positions in A and B after sorting
        total_operations = 0
        for a, b in zip(positions_A, positions_B):
            total_operations += abs(a - b)
        
        results.append(str(total_operations))
    
    print('\n'.join(results))

solve()
