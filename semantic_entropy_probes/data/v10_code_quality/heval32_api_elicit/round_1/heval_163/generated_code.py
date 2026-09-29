def generate_integers(a, b):
    """
    Given two positive integers a and b, return the even numbers between a
    and b. It means the even NUMBERS themselves in the range (i.e. 2,4,6,8 as clamped),
    not individual digit characters extracted from multi-digit numbers.
    """
    import numpy as np
    return np.arange(np.ceil(a / 2) * 2, b + 1, 2)
