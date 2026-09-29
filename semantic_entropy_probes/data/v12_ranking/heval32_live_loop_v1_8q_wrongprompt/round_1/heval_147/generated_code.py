def get_max_triples(n):
    import math
    count = 0
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            for k in range(j + 1, n + 1):
                a = i * i - i + 1
                b = j * j - j + 1
                c = k * k - k + 1
                if (a + b + c) % 3 == 0:
                    count += 1
    return count
