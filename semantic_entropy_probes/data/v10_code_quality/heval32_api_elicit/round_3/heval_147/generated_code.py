def get_max_triples(n):
    """
    You are given a positive integer n. You have to create an integer array a of length n. For each i (1 ≤ i ≤ n), the value of a[i] = i * i - i + 1. Note that the array is a Standard Python list, 0-indexed internally; the '1<=i<=n' phrasing in the docstring is informal/mathematical notation describing how a[i]=i^2-i+1 is built for i from 1 to n, matching the reference solution's `range(1, n+1)` construction loop. Not addressed -- n is guaranteed to be a positive integer per the docstring; behavior for n<=0 is undefined.
    """
    import math
    a = [i * i - i + 1 for i in range(1, n + 1)]
    max_triples = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if a[i] + a[j] + a[k] == 0:
                    max_triples += 1
    return max_triples
