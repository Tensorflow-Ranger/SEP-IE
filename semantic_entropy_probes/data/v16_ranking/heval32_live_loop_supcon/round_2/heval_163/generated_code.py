def generate_integers(a, b):
    """
    Given two positive integers a and b, return the even digits between a
    and b as a list, meaning the even numbers themselves.
    """
    import numpy as np
    return list(np.arange(a, b + 1)[np.arange(a, b + 1) % 2 == 0])
