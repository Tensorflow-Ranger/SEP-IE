import sys
from itertools import combinations

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    total = sum(A)
    
    # The XOR of all final B_i is the same as the XOR of all A_i, because moving stones doesn't change the total
    # So the XOR of all B_i is fixed as total_xor = A[0] ^ A[1] ^ ... ^ A[N-1]
    total_xor = 0
    for a in A:
        total_xor ^= a
    
    # The problem reduces to finding the number of distinct XOR values of subsets of A
    # Because we can choose which bags to merge into others, the final XOR is the XOR of some subset of A
    # (since the rest will be merged into others and their XOR will cancel out)
    
    # Generate all possible subset XORs
    subset_xors = set()
    for mask in range(1 << N):
        xor = 0
        for i in range(N):
            if mask & (1 << i):
                xor ^= A[i]
        subset_xors.add(xor)
    
    # The answer is the number of distinct subset XORs
    print(len(subset_xors))

if __name__ == "__main__":
    main()
