# Read input
s1, s2 = input().split()

# Check if both are fine
if s1 == 'fine' and s2 == 'fine':
    print(4)
# Check if both are sick
elif s1 == 'sick' and s2 == 'sick':
    print(1)
# Otherwise, determine based on who is sick
else:
    if s1 == 'sick':
        print(2)
    else:
        print(3)
