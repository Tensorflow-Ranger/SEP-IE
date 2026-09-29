def get_max_triples(n):
    """
    You are given a positive integer n. You have to create an integer array a of length n.
        For each i (1 ≤ i ≤ n), the value of a[i] = i * i - i + 1.
    """
    import math
    max_triples = 0
    for i in range(1, n + 1):
        a_i = i * i - i + 1
        for j in range(i + 1, n + 1):
            a_j = j * j - j + 1
            for k in range(j + 1, n + 1):
                a_k = k * k - k + 1
                if a_i + a_j + a_k == 0:
                    max_triples += 1
    return max_triples
