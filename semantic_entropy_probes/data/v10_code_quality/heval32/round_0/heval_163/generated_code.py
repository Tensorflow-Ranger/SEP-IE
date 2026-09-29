def generate_integers(a, b):
    """
    Given two positive integers a and b, return the even digits between a
    and b.
    """
    import numpy as np
    return np.arange(a, b+1)[np.char.mod('%d', np.arange(a, b+1)) % 2 == 0]
