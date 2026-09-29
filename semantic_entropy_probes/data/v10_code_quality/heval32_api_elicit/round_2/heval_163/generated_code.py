def generate_integers(a, b):
    import itertools
    return list(itertools.filterfalse(lambda x: x % 2 != 0, range(min(a, b), max(a, b) + 1)))
