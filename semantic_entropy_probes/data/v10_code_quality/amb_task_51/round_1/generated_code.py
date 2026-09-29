# Read input
A = list(map(int, input().split()))

# Check all permutations of A
from itertools import permutations

for perm in permutations(A):
    if perm[0] * perm[1] == perm[2]:
        print("Yes")
        break
else:
    print("No")
