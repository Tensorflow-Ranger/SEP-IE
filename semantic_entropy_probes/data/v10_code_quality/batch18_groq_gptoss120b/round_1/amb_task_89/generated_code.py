import sys

def solve() -> None:
    data = sys.stdin.read().strip().split()
    if len(data) != 2:
        return
    s1, s2 = data
    if s1 == "fine" and s2 == "fine":
        print(4)
    elif s1 == "sick" and s2 == "sick":
        print(1)
    elif s1 == "sick" and s2 == "fine":
        print(2)
    else:  # s1 == "fine" and s2 == "sick"
        print(3)

if __name__ == "__main__":
    solve()
