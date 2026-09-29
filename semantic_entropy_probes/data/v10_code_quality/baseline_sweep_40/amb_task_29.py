import sys
import math

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = data[1]

    length = 3 ** N
    target = '0' if A.count('1') < length // 2 else '1'

    def min_changes_to_flip(A, target):
        n = len(A)
        changes = 0
        for i in range(0, n, 3):
            group = A[i:i+3]
            count = group.count(target)
            if count < 2:
                changes += 3 - count
        return changes

    print(min_changes_to_flip(A, target))

if __name__ == "__main__":
    main()
