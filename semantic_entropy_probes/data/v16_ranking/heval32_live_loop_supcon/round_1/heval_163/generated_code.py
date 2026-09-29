def generate_integers(a, b):
    """
    Given two positive integers a and b, return the even digits between a
    and b as a list.
    """
    import numpy as np
    return list(np.arange(a, b + 1)[np.char.mod('%d', np.arange(a, b + 1)) % 2 == 0])
