# Read input
S1, S2 = input().split()

# Determine which oysters could cause sickness
possible_causes = []

# Check if Takahashi got sick
if S1 == 'sick':
    possible_causes.extend([1, 2])

# Check if Aoki got sick
if S2 == 'sick':
    possible_causes.extend([1, 3])

# Remove duplicates
possible_causes = list(set(possible_causes))

# If both are fine, oyster 4 is the cause
if S1 == 'fine' and S2 == 'fine':
    print(4)
else:
    print(possible_causes[0])
