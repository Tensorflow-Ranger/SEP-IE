S1, S2 = input().split()

# Takahashi ate oysters 1 and 2
# Aoki ate oysters 1 and 3

if S1 == "sick" and S2 == "sick":
    # Both sick - the common oyster (1) must be the culprit
    print(1)
elif S1 == "sick" and S2 == "fine":
    # Only Takahashi sick - he ate 1 and 2, Aoki ate 1 and 3 but is fine
    # So 2 must be the culprit
    print(2)
elif S1 == "fine" and S2 == "sick":
    # Only Aoki sick - he ate 1 and 3, Takahashi ate 1 and 2 but is fine
    # So 3 must be the culprit
    print(3)
else:
    # Both fine - the one neither ate (4) must be the culprit
    print(4)
