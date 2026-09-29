import sys

def solve(S):
    # Keep replacing "WA" with "AC" until no more "WA" exists
    while "WA" in S:
        # Replace the first occurrence of "WA" with "AC"
        S = S.replace("WA", "AC", 1)
    return S

# Read input
S = sys.stdin.readline().strip()

# Solve and print result
result = solve(S)
print(result)
