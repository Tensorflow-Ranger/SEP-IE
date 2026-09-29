# Read input
S1, S2 = input().split()

# Determine which oyster causes stomach trouble
if S1 == 'fine' and S2 == 'fine':
    print(4)
elif S1 == 'sick' and S2 == 'fine':
    print(2)
elif S1 == 'fine' and S2 == 'sick':
    print(3)
else:
    print(1)
