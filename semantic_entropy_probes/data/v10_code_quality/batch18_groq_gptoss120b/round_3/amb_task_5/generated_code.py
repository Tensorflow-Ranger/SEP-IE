import sys

def main():
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    total_masks = 1 << N
    # precompute sums of subsets
    subset_sum = [0] * total_masks
    for mask in range(1, total_masks):
        lsb = mask & -mask
        idx = (lsb.bit_length() - 1)
        subset_sum[mask] = subset_sum[mask ^ lsb] + A[idx]

    dp = [set() for _ in range(total_masks)]
    dp[0].add(0)

    for mask in range(1, total_masks):
        sub = mask
        while sub:
            rest = mask ^ sub
            s = subset_sum[sub]
            for val in dp[rest]:
                dp[mask].add(val ^ s)
            sub = (sub - 1) & mask

    full = total_masks - 1
    print(len(dp[full]))

if __name__ == "__main__":
    main()
