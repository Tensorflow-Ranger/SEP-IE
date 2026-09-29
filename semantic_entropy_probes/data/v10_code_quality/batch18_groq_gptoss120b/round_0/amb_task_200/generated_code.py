import sys

def main() -> None:
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    B = [int(next(it)) for _ in range(N)]
    W = [int(next(it)) for _ in range(M)]

    B.sort(reverse=True)
    W.sort(reverse=True)

    # prefix sums
    prefB = [0] * (N + 1)
    prefW = [0] * (M + 1)
    for i in range(N):
        prefB[i + 1] = prefB[i] + B[i]
    for i in range(M):
        prefW[i + 1] = prefW[i] + W[i]

    # prefix sums of positive black values
    pos_pref = [0] * (N + 1)
    for i in range(N):
        pos_pref[i + 1] = pos_pref[i] + (B[i] if B[i] > 0 else 0)
    total_positive = pos_pref[N]

    limit = min(N, M)
    ans = -10**30  # sufficiently small
    for w in range(limit + 1):
        cur = prefW[w] + prefB[w] + (total_positive - pos_pref[w])
        if cur > ans:
            ans = cur

    # also consider taking only black balls (w=0 already covers)
    print(ans)

if __name__ == "__main__":
    main()
