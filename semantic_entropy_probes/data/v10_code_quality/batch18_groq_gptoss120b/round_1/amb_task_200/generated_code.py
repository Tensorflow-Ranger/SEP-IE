import sys

def solve() -> None:
    data = list(map(int, sys.stdin.read().strip().split()))
    if not data:
        return
    it = iter(data)
    N = next(it)
    M = next(it)
    blacks = [next(it) for _ in range(N)]
    whites = [next(it) for _ in range(M)]

    pos_blacks = [b for b in blacks if b > 0]
    pos_whites = [w for w in whites if w > 0]

    sum_black = sum(pos_blacks)
    cnt_black = len(pos_blacks)

    sum_white = sum(pos_whites)
    cnt_white = len(pos_whites)

    if cnt_white > cnt_black:
        pos_whites.sort()                     # ascending
        excess = cnt_white - cnt_black
        # drop the smallest 'excess' whites
        drop = sum(pos_whites[:excess])
        sum_white -= drop
        cnt_white -= excess

    print(sum_black + sum_white)

if __name__ == "__main__":
    solve()
