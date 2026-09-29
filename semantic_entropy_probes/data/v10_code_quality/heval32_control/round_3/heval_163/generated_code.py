def generate_integers(a, b):
    import math
    start = max(2, min(a, b))
    end = min(8, max(a, b))
    return list(range(start, end + 1, 2))
