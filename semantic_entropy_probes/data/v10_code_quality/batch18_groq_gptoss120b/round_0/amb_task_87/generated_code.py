import sys

def main():
    data = sys.stdin.read().strip().split()
    if len(data) != 2:
        return
    S1, S2 = data
    for x in range(1, 5):
        takahashi_sick = "sick" if x in (1, 2) else "fine"
        aoki_sick = "sick" if x in (1, 3) else "fine"
        if takahashi_sick == S1 and aoki_sick == S2:
            print(x)
            return

if __name__ == "__main__":
    main()
