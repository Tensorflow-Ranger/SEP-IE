import sys

def main():
    data = sys.stdin.read().strip().split()
    if len(data) != 2:
        return
    S1, S2 = data
    takahashi = {1, 2}
    aoki = {1, 3}
    for bad in range(1, 5):
        ok = True
        if S1 == "sick":
            ok &= bad in takahashi
        else:  # fine
            ok &= bad not in takahashi
        if S2 == "sick":
            ok &= bad in aoki
        else:
            ok &= bad not in aoki
        if ok:
            print(bad)
            return

if __name__ == "__main__":
    main()
