import sys

def solve() -> None:
    data = sys.stdin.read().strip().split()
    if len(data) < 2:
        return
    s1, s2 = data[0], data[1]

    if s1 == "fine" and s2 == "fine":
        ans = 4
    elif s1 == "sick" and s2 == "sick":
        ans = 1
    elif s1 == "sick" and s2 == "fine":
        ans = 2
    else:  # s1 == "fine" and s2 == "sick"
        ans = 3

    print(ans)

if __name__ == "__main__":
    solve()
