# Read input
N, M = map(int, input().split())

# Handle the case when N == 1
if N == 1:
    print(M + 1)
else:
    # Calculate the sum of geometric series
    result = (N ** (M + 1) - 1) // (N - 1)
    print(result)
