import sys
from itertools import combinations

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    total_xor = 0
    for a in A:
        total_xor ^= a
    
    result = set()
    
    for mask in range(1, 1 << N):
        xor_sum = 0
        for i in range(N):
            if mask & (1 << i):
                xor_sum ^= A[i]
        result.add(xor_sum)
    
    print(len(result))

if __name__ == "__main__":
    main()
