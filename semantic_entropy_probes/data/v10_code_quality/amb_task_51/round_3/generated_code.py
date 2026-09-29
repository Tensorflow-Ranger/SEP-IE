# Read input
a = list(map(int, input().split()))

# Check all permutations of the list
from itertools import permutations

for perm in permutations(a):
    if perm[0] * perm[1] == perm[2]:
        print("Yes")
        break
else:
    print("No")
