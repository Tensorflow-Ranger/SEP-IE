# Read input
S1, S2 = input().split()

# Determine which oyster types cause sickness based on the symptoms
# Takahashi ate oysters 1 and 2
# Aoki ate oysters 1 and 3

# Check for possible causes
if S1 == 'sick' and S2 == 'sick':
    # Both got sick, so oyster 1 must be the cause
    print(1)
elif S1 == 'sick' and S2 == 'fine':
    # Only Takahashi got sick, so oyster 2 must be the cause
    print(2)
elif S1 == 'fine' and S2 == 'sick':
    # Only Aoki got sick, so oyster 3 must be the cause
    print(3)
else:
    # Both are fine, so oyster 4 must be the cause
    print(4)
