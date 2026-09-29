S_1, S_2 = input().split()

if S_1 == "sick" and S_2 == "sick":
    print(1)
elif S_1 == "sick" and S_2 == "fine":
    print(2)
elif S_1 == "fine" and S_2 == "sick":
    print(3)
else:
    print(4)
