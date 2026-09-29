import sys

def solve() -> None:
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    R = int(data[1])
    C = int(data[2])
    S = data[3].strip()

    r = c = 0
    out = []
    for ch in S:
        if ch == 'N':
            r -= 1
        elif ch == 'S':
            r += 1
        elif ch == 'W':
            c -= 1
        else:  # 'E'
            c += 1
        out.append('1' if (r == R and c == C) else '0')
    sys.stdout.write(''.join(out))

if __name__ == "__main__":
    solve()
