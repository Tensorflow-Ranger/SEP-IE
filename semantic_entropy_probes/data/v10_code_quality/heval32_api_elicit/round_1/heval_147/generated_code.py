def get_max_triples(n):
    """
    You are given a positive integer n. You have to create an integer array a of length n.
        For each i (1 ≤ i ≤ n), the value of a[i] = i * i - i + 1. None; the function is a pure computation with no printing or global-state modification.
    """
    import math
    a = [i * i - i + 1 for i in range(1, n + 1)]
    max_triples = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if math.gcd(a[i], a[j]) == 1 and math.gcd(a[j], a[k]) == 1 and math.gcd(a[i], a[k]) == 1:
                    max_triples += 1
    return max_triples
