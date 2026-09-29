N, M = map(int, input().split())

if N == 1:
    print(M + 1)
else:
    result = (N ** (M + 1) - 1) // (N - 1)
    print(result)
