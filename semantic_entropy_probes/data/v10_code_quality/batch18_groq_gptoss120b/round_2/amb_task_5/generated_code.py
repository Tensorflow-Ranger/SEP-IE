import sys

def solve() -> None:
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    full = (1 << N) - 1

    # precompute sum for each mask
    sums = [0] * (1 << N)
    for mask in range(1, 1 << N):
        lsb = mask & -mask
        idx = (lsb.bit_length() - 1)
        sums[mask] = sums[mask ^ lsb] + A[idx]

    dp = [set() for _ in range(1 << N)]
    dp[0].add(0)

    for mask in range(1, 1 << N):
        sub = mask
        while sub:
            rest = mask ^ sub
            s = sums[sub]
            for val in dp[rest]:
                dp[mask].add(val ^ s)
            sub = (sub - 1) & mask

    print(len(dp[full]))

if __name__ == "__main__":
    solve()
