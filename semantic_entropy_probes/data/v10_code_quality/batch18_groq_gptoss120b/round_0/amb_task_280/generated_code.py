import sys

def solve() -> None:
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    R = int(data[1])
    C = int(data[2])
    S = data[3].strip()

    # prefix positions P[0..N]
    px = [0] * (N + 1)
    py = [0] * (N + 1)
    for i, ch in enumerate(S, 1):
        dx = dy = 0
        if ch == 'N':
            dx = -1
        elif ch == 'S':
            dx = 1
        elif ch == 'W':
            dy = -1
        else:  # 'E'
            dy = 1
        px[i] = px[i-1] + dx
        py[i] = py[i-1] + dy

    start_set = {(0, 0)}          # contains P[s] for existing start times
    out = []

    for t in range(1, N + 1):
        need = (px[t] - R, py[t] - C)
        out.append('1' if need in start_set else '0')

        cur = (px[t], py[t])
        if cur not in start_set:   # origin empty → new particle appears
            start_set.add(cur)

    sys.stdout.write(''.join(out))

if __name__ == "__main__":
    solve()
