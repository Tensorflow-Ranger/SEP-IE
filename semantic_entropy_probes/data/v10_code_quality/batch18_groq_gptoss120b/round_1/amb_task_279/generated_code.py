import sys

def solve() -> None:
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    R = int(data[1])
    C = int(data[2])
    S = data[3].strip()

    # displacement after i steps
    dx = dy = 0

    # first occurrence of each prefix pair
    first = {(0, 0): 0}

    out = []
    for t, ch in enumerate(S, 1):
        if ch == 'N':
            dx -= 1
        elif ch == 'S':
            dx += 1
        elif ch == 'W':
            dy -= 1
        else:  # 'E'
            dy += 1

        need = (dx - R, dy - C)
        out.append('1' if need in first else '0')

        if (dx, dy) not in first:
            first[(dx, dy)] = t

    sys.stdout.write(''.join(out))

if __name__ == "__main__":
    solve()
