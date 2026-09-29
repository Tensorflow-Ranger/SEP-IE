# Read input
A = list(map(int, input().split()))

# Generate all permutations of A
from itertools import permutations

# Check if any permutation satisfies B1 * B2 == B3
for perm in permutations(A):
    if perm[0] * perm[1] == perm[2]:
        print("Yes")
        break
else:
    print("No")
