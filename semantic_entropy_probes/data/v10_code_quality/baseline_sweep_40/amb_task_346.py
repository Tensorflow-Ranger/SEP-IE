import sys
input = sys.stdin.read

def main():
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
        
        # Convert A and B to lists of integers
        A = [int(c) for c in A]
        B = [int(c) for c in B]
        
        # Check if the target configuration is possible
        # For each position i, if B[i] == 1, there must be at least one piece in the target configuration
        # Also, the number of 1s in A and B must be the same
        count_A = sum(A)
        count_B = sum(B)
        if count_A != count_B:
            results.append("-1")
            continue
        
        # For each position i, if B[i] == 1, we need to have a piece there
        # We will track the positions of the pieces in A and the required positions in B
        # We will simulate the movement of pieces to reach the target configuration
        
        # Find the positions of 1s in A
        positions_A = [i for i in range(N) if A[i] == 1]
        # Find the positions of 1s in B
        positions_B = [i for i in range(N) if B[i] == 1]
        
        # We need to match the positions in positions_A to positions_B
        # The optimal way is to match the i-th piece in A to the i-th position in B
        # This is because moving a piece from position a to position b takes |a - b| steps
        # and we want to minimize the total steps
        
        total_moves = 0
        for a, b in zip(positions_A, positions_B):
            total_moves += abs(a - b)
        
        results.append(str(total_moves))
    
    print("\n".join(results))

if __name__ == "__main__":
    main()
