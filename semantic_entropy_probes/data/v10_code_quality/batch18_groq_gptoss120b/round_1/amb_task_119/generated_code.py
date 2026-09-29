import sys

def solve() -> None:
    s = sys.stdin.readline().strip()
    n = len(s)
    res = []
    i = 0
    while i < n:
        if s[i] == 'W' and i + 1 < n and s[i + 1] == 'A':
            res.append('A')
            res.append('C')
            i += 2
        else:
            res.append(s[i])
            i += 1
    sys.stdout.write(''.join(res))

if __name__ == "__main__":
    solve()
