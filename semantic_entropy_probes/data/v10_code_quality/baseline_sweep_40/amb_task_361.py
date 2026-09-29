import sys
import sys
import math
from collections import defaultdict

MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    s = data[1]

    # For each vertex, we need to count how many edges can be directed towards it
    # We'll use dynamic programming to count the number of valid configurations

    # dp[i][j] = number of ways to assign directions to the first i edges such that the total in-degree of vertex N is j
    # But since N is up to 1e6, we need a more efficient approach

    # Let's think about the in-degrees of the vertices
    # For vertex 0, it can receive edges from:
    # - edge between 0 and 1 (directed from 1 to 0)
    # - edge between N and 0 (directed from N to 0) if s[0] is 1
    # Similarly for other vertices

    # The in-degree of vertex i is:
    # - 1 if the edge between i and (i+1) mod N is directed towards i
    # - 1 if the edge between i and N is directed towards i and s[i] is 1
    # So for each vertex, the in-degree can be 0, 1, or 2

    # However, the total number of configurations is 2^(number of edges)
    # But we need to count the number of distinct sequences of in-degrees

    # Let's count the number of edges
    # There are N edges between consecutive vertices (circular)
    # And N edges between vertex N and each vertex i if s[i] is 1
    # So total edges = N + sum(s)

    # The number of configurations is 2^(number of edges)
    # But we need to count the number of distinct sequences of in-degrees

    # However, the in-degrees of the vertices are not independent
    # So we need to find the number of distinct (d_0, d_1, ..., d_N) sequences

    # Let's think about the in-degrees of the vertices
    # For vertex i (0 <= i < N):
    # - It can receive 1 from the edge between i and (i+1) mod N
    # - It can receive 1 from the edge between i and N if s[i] is 1
    # So the in-degree of vertex i can be 0, 1, or 2

    # For vertex N:
    # - It can receive 1 from each edge between i and N if s[i] is 1
    # So the in-degree of vertex N is equal to the number of 1s in s

    # So the in-degree of vertex N is fixed and equal to the number of 1s in s
    # So the only variable in-degrees are for vertices 0 to N-1

    # So the number of distinct sequences is equal to the number of distinct (d_0, d_1, ..., d_{N-1}) sequences
    # where each d_i can be 0, 1, or 2, and the sum of d_i is equal to the number of edges directed towards the vertices 0 to N-1

    # The number of edges is N + sum(s)
    # Each edge contributes to exactly one in-degree (either towards a vertex or not)
    # So the total number of in-degrees for vertices 0 to N-1 is equal to the number of edges

    # So the problem reduces to counting the number of distinct (d_0, d_1, ..., d_{N-1}) sequences
    # where each d_i is 0, 1, or 2, and the sum of d_i is equal to the number of edges

    # Let's compute the number of edges
    num_edges = N + sum(int(c) for c in s)

    # Now, we need to count the number of sequences (d_0, d_1, ..., d_{N-1}) where each d_i is 0, 1, or 2, and sum(d_i) = num_edges

    # This is a classic integer composition problem with constraints
    # We can use dynamic programming to count the number of ways to assign values to d_i such that the sum is num_edges

    # Let dp[i][j] be the number of ways to assign values to the first i vertices such that the sum is j
    # The recurrence is:
    # dp[i][j] = dp[i-1][j] + dp[i-1][j-1] + dp[i-1][j-2]

    # But since N can be up to 1e6, we need to optimize this

    # Let's use a 1D DP array
    # dp[j] = number of ways to get sum j with the first i vertices
    # We'll use a rolling array

    max_sum = num_edges
    dp = [0] * (max_sum + 1)
    dp[0] = 1

    for i in range(N):
        new_dp = [0] * (max_sum + 1)
        for j in range(max_sum + 1):
            if dp[j]:
                new_dp[j] = (new_dp[j] + dp[j]) % MOD
                if j + 1 <= max_sum:
                    new_dp[j + 1] = (new_dp[j + 1] + dp[j]) % MOD
                if j + 2 <= max_sum:
                    new_dp[j + 2] = (new_dp[j + 2] + dp[j]) % MOD
        dp = new_dp

    # The answer is dp[num_edges]
    print(dp[num_edges] % MOD)

if __name__ == "__main__":
    main()
