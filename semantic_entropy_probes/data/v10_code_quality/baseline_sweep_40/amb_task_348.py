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
        
        # Precompute positions of 1s in A
        A_pos = [i for i in range(N) if A[i] == '1']
        B_pos = [i for i in range(N) if B[i] == '1']
        
        # Check if B has at least one 1
        if not B_pos:
            results.append(-1)
            continue
        
        # Check if the number of 1s in A and B are the same
        if len(A_pos) != len(B_pos):
            results.append(-1)
            continue
        
        # Sort positions for both A and B
        A_pos.sort()
        B_pos.sort()
        
        # Check if the positions can be matched with some shift
        # We need to find a shift such that for each i, A_pos[i] + shift = B_pos[i]
        # So shift = B_pos[i] - A_pos[i] for all i
        # All shifts must be the same
        shift = B_pos[0] - A_pos[0]
        valid = True
        for i in range(1, len(A_pos)):
            if B_pos[i] - A_pos[i] != shift:
                valid = False
                break
        
        if not valid:
            results.append(-1)
            continue
        
        # Now compute the total number of operations
        # Each piece moves from A_pos[i] to B_pos[i]
        # The number of operations is the sum of absolute differences between A_pos and B_pos
        total_ops = 0
        for a, b in zip(A_pos, B_pos):
            total_ops += abs(b - a)
        
        results.append(total_ops)
    
    for res in results:
        print(res)

solve()
