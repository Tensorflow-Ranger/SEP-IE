def generate_integers(a, b):
    """
    Given two positive integers a and b, return the even digits between a
    and b. 'Even digits' refers to the actual digit values 0, 2, 4, 6, 8 (i.e. even single digits, not even multi-digit numbers) -- specifically these digit values that fall within the inclusive range [a, b], returned in ascending order; digits like 0 that may be outside [a,b] (since a, b are positive integers >= 1) are naturally excluded by the range check.
    """
    import itertools

    def is_even_digit(n):
        return n in {0, 2, 4, 6, 8}

    result = []
    for num in range(a, b + 1):
        for digit in str(num):
            if is_even_digit(int(digit)):
                result.append(int(digit))
    return sorted(set(result))
