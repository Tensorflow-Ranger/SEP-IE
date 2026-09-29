S1, S2 = input().split()

# Takahashi ate oysters 1 and 2
# Aoki ate oysters 1 and 3

# If someone gets sick, at least one of the oysters they ate causes trouble
# If someone doesn't get sick, none of the oysters they ate cause trouble

# Let's analyze all cases:
if S1 == "fine" and S2 == "fine":
    # Both are fine, so oysters 1, 2, and 3 are safe
    # Therefore, oyster 4 causes trouble
    print(4)
elif S1 == "sick" and S2 == "fine":
    # Takahashi sick, Aoki fine
    # Aoki fine means oysters 1 and 3 are safe
    # Takahashi sick means at least one of 1 or 2 causes trouble
    # Since 1 is safe (from Aoki), 2 must cause trouble
    print(2)
elif S1 == "fine" and S2 == "sick":
    # Takahashi fine, Aoki sick
    # Takahashi fine means oysters 1 and 2 are safe
    # Aoki sick means at least one of 1 or 3 causes trouble
    # Since 1 is safe (from Takahashi), 3 must cause trouble
    print(3)
else:  # S1 == "sick" and S2 == "sick"
    # Both sick
    # Takahashi sick means at least one of 1 or 2 causes trouble
    # Aoki sick means at least one of 1 or 3 causes trouble
    # If oyster 1 caused trouble, both would be sick - this is possible
    # But let's check if we can determine uniquely:
    # If 1 is safe, then 2 causes trouble (from Takahashi) and 3 causes trouble (from Aoki)
    # But that would mean two oysters cause trouble, which might not be intended
    # However, the problem states "one or some types cause stomach trouble"
    # So multiple types can cause trouble
    # But we need to find THE label that causes trouble
    # Actually, looking more carefully at the problem:
    # "If both S_1 and S_2 are 'fine', then oysters 1, 2, and 3 are all safe, so oyster 4 is the one causing stomach trouble."
    # This suggests there's exactly one oyster causing trouble in any case
    # So in the case both are sick, oyster 1 must be the culprit
    print(1)
