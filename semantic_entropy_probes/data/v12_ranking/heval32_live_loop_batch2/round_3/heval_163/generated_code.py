def generate_integers(a, b):
    """
    Given two positive integers a and b, return the even digits between a
    and b. By 'even digits' the specification means the even single-digit numbers themselves (2, 4, 6, 8) that fall within the range [a, b], not individual digits extracted from multi-digit numbers. The number 0 is not returned as it is not included in the set of even digits (2, 4, 6, 8); the minimum is 2. Return the even digits in ascending order; the function automatically handles the range correctly by using min/max of a and b.
    """
    start = min(a, b)
    end = max(a, b)
    even_digits = [i for i in range(start, end + 1) if i in {2, 4, 6, 8}]
    return even_digits
