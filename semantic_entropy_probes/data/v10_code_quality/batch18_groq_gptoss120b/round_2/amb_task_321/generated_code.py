import sys

def solve() -> None:
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N, M = map(int, data[:2])
    if N == 1:
        result = M + 1
    else:
        # Use geometric series formula: (N^{M+1} - 1) / (N - 1)
        result = (pow(N, M + 1) - 1) // (N - 1)
    sys.stdout.write(str(result))

if __name__ == "__main__":
    solve()
