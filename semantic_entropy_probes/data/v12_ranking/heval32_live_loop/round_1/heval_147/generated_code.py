def get_max_triples(n):
    """
    You are given a positive integer n. You have to create an integer array a of length n.
        For each i (1 ≤ i ≤ n), the value of a[i] = i * i - i + 1. The function returns the count (an integer) of valid triples, not the array itself.
    """
    count = 0
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            for k in range(1, n + 1):
                if i != j and j != k and i != k:
                    a_i = i * i - i + 1
                    a_j = j * j - j + 1
                    a_k = k * k - k + 1
                    if a_i + a_j + a_k == 0:
                        count += 1
    return count
