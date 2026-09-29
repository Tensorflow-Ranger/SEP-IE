S1, S2 = input().split()

# Takahashi ate oysters 1 and 2
# Aoki ate oysters 1 and 3

# If Takahashi is sick, then at least one of oysters 1 or 2 is bad
# If Takahashi is fine, then both oysters 1 and 2 are good
# If Aoki is sick, then at least one of oysters 1 or 3 is bad
# If Aoki is fine, then both oysters 1 and 3 are good

bad_oysters = set()

if S1 == "sick":
    # At least one of oysters 1 or 2 is bad
    pass
else:  # S1 == "fine"
    # Both oysters 1 and 2 are good
    bad_oysters.add(1)
    bad_oysters.add(2)

if S2 == "sick":
    # At least one of oysters 1 or 3 is bad
    pass
else:  # S2 == "fine"
    # Both oysters 1 and 3 are good
    bad_oysters.add(1)
    bad_oysters.add(3)

# Now we know which oysters are definitely good
# The bad oyster is the one that's not in bad_oysters

all_oysters = {1, 2, 3, 4}
good_oysters = all_oysters - bad_oysters

# Let's think differently
# We need to find which oyster(s) could be bad based on the constraints

possible_bad = set()

# Check each oyster to see if it could be the bad one
for bad in range(1, 5):
    # Check if this assumption is consistent with observations
    
    # If oyster 'bad' is bad, then:
    # Takahashi gets sick if he ate oyster 'bad' (i.e., if bad is 1 or 2)
    takahashi_sick = (bad == 1 or bad == 2)
    
    # Aoki gets sick if he ate oyster 'bad' (i.e., if bad is 1 or 3)
    aoki_sick = (bad == 1 or bad == 3)
    
    # Check if this matches the input
    if (takahashi_sick == (S1 == "sick")) and (aoki_sick == (S2 == "sick")):
        possible_bad.add(bad)

# The problem seems to imply there's exactly one bad oyster
# Let's check all cases systematically

if S1 == "fine" and S2 == "fine":
    # Both are fine, so oysters 1, 2, and 3 are all good
    # Only oyster 4 could be bad
    print(4)
elif S1 == "fine" and S2 == "sick":
    # Takahashi fine (so 1 and 2 are good), Aoki sick
    # Aoki ate 1 (good) and 3, so 3 must be bad
    print(3)
elif S1 == "sick" and S2 == "fine":
    # Takahashi sick, Aoki fine (so 1 and 3 are good)
    # Takahashi ate 1 (good) and 2, so 2 must be bad
    print(2)
else:  # S1 == "sick" and S2 == "sick"
    # Both sick
    # Takahashi ate 1 and 2, Aoki ate 1 and 3
    # If 1 is bad, both get sick
    print(1)
