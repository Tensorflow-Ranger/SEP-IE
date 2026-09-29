def generate_integers(a, b):
    import math
    lower = min(a, b)
    upper = max(a, b)
    return [i for i in range(lower, upper + 1) if i % 2 == 0]
