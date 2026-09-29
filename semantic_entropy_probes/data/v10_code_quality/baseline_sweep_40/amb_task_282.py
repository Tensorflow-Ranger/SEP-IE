import sys
from collections import defaultdict

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    count = defaultdict(int)
    for a in A:
        count[a] += 1
    
    unique = [a for a in A if count[a] == 1]
    if not unique:
        print(-1)
    else:
        max_val = max(unique)
        for i in range(N):
            if A[i] == max_val:
                print(i + 1)
                return

if __name__ == "__main__":
    main()
