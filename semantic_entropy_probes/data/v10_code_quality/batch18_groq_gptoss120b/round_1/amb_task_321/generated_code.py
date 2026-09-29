import sys

def solve() -> None:
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    M = int(data[1])
    if N == 1:
        result = M + 1
    else:
        # Compute N^(M+1) using pow with big integers
        result = (pow(N, M + 1) - 1) // (N - 1)
    sys.stdout.write(str(result))

if __name__ == "__main__":
    solve()
