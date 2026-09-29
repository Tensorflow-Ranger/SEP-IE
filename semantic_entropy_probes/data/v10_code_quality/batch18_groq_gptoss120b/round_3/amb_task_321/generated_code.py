N, M = map(int, input().split())

if N == 1:
    if M == -1:
        print("inf")
    else:
        print(M + 1)
else:
    if M == -1:
        print("inf")
    else:
        # Using geometric series formula: (N^(M+1) - 1) / (N - 1)
        result = (pow(N, M + 1) - 1) // (N - 1)
        print(result)
