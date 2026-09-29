def get_max_triples(n):
    import itertools
    a = [i*i - i + 1 for i in range(1, n+1)]
    count = 0
    for i, j, k in itertools.combinations(range(n), 3):
        if (a[i] + a[j] + a[k]) % 3 == 0:
            count += 1
    return count
