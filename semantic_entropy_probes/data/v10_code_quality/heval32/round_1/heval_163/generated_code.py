def generate_integers(a, b):
    """
    Given two positive integers a and b, return the even numbers between a
    and b. Despite the name, it means numbers that are themselves even (i % 2 == 0),
    not numbers composed of even digits; e.g. generate_integers(2,8) returns [2,4,6,8],
    the even NUMBERS in that range.
    """
    import numpy as np
    return np.arange(np.minimum(a, b), np.maximum(a, b) + 1, 2).tolist()
