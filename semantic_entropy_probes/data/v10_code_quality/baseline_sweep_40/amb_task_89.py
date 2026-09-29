# Read input
S1, S2 = input().split()

# Check each oyster type to see if it causes sickness
for cause in range(1, 5):
    # Takahashi ate oysters 1 and 2
    takahashi_sick = (1 == cause) or (2 == cause)
    # Aoki ate oysters 1 and 3
    aoki_sick = (1 == cause) or (3 == cause)
    
    # Check if the sickness matches the input
    if (S1 == 'sick' and takahashi_sick) and (S2 == 'sick' and aoki_sick):
        print(cause)
        break
