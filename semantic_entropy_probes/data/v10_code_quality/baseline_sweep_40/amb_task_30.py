import sys
from collections import Counter

def majority(s):
    cnt = Counter(s)
    return cnt.most_common(1)[0][0]

def apply_operation(s):
    n = len(s) // 3
    result = []
    for i in range(n):
        group = s[3*i:3*i+3]
        result.append(majority(group))
    return ''.join(result)

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = data[1]
    
    current = A
    for _ in range(N):
        current = apply_operation(current)
    
    # Now find the minimum number of changes needed to make current be '0' or '1'
    # and choose the minimum between the two
    if current == '0':
        print(0)
    elif current == '1':
        print(0)
    else:
        # This case should not happen as current is length 1
        print(0)

if __name__ == "__main__":
    main()
