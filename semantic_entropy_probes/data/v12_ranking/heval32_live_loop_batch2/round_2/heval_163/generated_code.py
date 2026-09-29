def generate_integers(a, b):
    """
    Given two positive integers a and b, return the even digits between a
    and b. By 'even digits' the specification means the even single-digit numbers themselves (2, 4, 6, 8) that fall within the range [a, b], not individual digits extracted from multi-digit numbers. Return the even digits in ascending order; the function automatically handles the range correctly by using min/max of a and b.
    """
    import numpy as np
    start = min(a, b)
    end = max(a, b)
    even_digits = np.array([2, 4, 6, 8])
    return even_digits[(even_digits >= start) & (even_digits <= end)].tolist()
