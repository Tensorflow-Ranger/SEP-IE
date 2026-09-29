import sys
from math import log, pow

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = data[1]

    length = 3 ** N
    target = 1 - int(A[0])  # We want to flip the final result

    # Precompute powers of 3
    pow3 = [3 ** i for i in range(N + 1)]

    # Function to apply the operation once
    def apply_operation(s):
        new_s = []
        for i in range(0, len(s), 3):
            group = s[i:i+3]
            count = group.count('1')
            if count >= 2:
                new_s.append('1')
            else:
                new_s.append('0')
        return ''.join(new_s)

    # We need to find the minimum number of changes to make the final result different
    # We can simulate the process in reverse to find the minimal changes
    # Start from the final result and work backwards

    # Initialize the result string
    result = '1' if target == 1 else '0'

    for level in range(N, 0, -1):
        new_result = []
        for i in range(0, len(result), 3):
            group = result[i:i+3]
            count = group.count('1')
            if count >= 2:
                new_result.append('1')
            else:
                new_result.append('0')
        result = ''.join(new_result)

    # Now, we need to find the minimal number of changes to A to get to this result
    # We simulate the process forward and track the number of changes needed
    current = A
    changes = 0
    for _ in range(N):
        new_current = []
        for i in range(0, len(current), 3):
            group = current[i:i+3]
            count = group.count('1')
            if count >= 2:
                new_current.append('1')
            else:
                new_current.append('0')
        current = ''.join(new_current)
        # Compare with the target result and count changes
        for i in range(len(current)):
            if current[i] != result[i]:
                changes += 1
    print(changes)

if __name__ == "__main__":
    main()
