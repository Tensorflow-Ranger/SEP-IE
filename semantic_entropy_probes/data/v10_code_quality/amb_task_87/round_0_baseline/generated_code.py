# Read input
S1, S2 = input().split()

# Determine which oysters caused sickness
takahashi_oysters = {1, 2}
aoki_oysters = {1, 3}

# Check if Takahashi got sick
takahashi_sick = S1 == 'sick'
aoki_sick = S2 == 'sick'

# Find the common oyster that caused sickness
common_oysters = takahashi_oysters & aoki_oysters

# The answer is the oyster that is in both sets and caused sickness
for oyster in common_oysters:
    if takahashi_sick and aoki_sick:
        print(oyster)
        break
