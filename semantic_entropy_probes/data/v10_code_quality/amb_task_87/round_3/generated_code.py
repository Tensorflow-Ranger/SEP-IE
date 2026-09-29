# Read input
S1, S2 = input().split()

# Determine which oyster types could cause sickness based on S1 and S2
# Oyster 1: eaten by both
# Oyster 2: eaten by Takahashi
# Oyster 3: eaten by Aoki
# Oyster 4: eaten by neither

# If both got sick, then oyster 1 is the cause
if S1 == 'sick' and S2 == 'sick':
    print(1)
# If only Takahashi got sick, then oyster 2 is the cause
elif S1 == 'sick' and S2 == 'fine':
    print(2)
# If only Aoki got sick, then oyster 3 is the cause
elif S1 == 'fine' and S2 == 'sick':
    print(3)
# If neither got sick, then oyster 4 is the cause
else:
    print(4)
