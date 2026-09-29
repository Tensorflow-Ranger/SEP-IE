# Read input
s1, s2 = input().split()

# Determine the culprit based on the given conditions
if s1 == 'fine' and s2 == 'fine':
    print(4)
elif s1 == 'sick' and s2 == 'sick':
    print(1)
elif s1 == 'sick' and s2 == 'fine':
    print(2)
else:  # s1 == 'fine' and s2 == 'sick'
    print(3)
