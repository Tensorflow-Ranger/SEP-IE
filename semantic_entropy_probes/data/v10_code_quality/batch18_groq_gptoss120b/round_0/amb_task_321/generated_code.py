import sys

def solve() -> None:
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    M = int(data[1])
    LIMIT = 10**18

    if N == 1:
        X = M + 1
        print(X if X <= LIMIT else "inf")
        return

    # compute N^{M+1} with early stop if exceeds LIMIT*(N-1)+1
    # because X = (N^{M+1}-1)//(N-1)
    # we need to know if X > LIMIT
    # So we can stop when numerator exceeds LIMIT*(N-1)+1
    thresh = LIMIT * (N - 1) + 1
    power = 1
    for _ in range(M + 1):
        if power > thresh:
            print("inf")
            return
        power *= N
    # now power = N^{M+1}
    X = (power - 1) // (N - 1)
    print(X if X <= LIMIT else "inf")

if __name__ == "__main__":
    solve()
