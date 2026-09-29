def is_equal_to_sum_even(n):
    """Evaluate whether the given number n can be written as the sum of some positive even numbers"""
    import math
    if n < 2:
        return False
    return (n % 2 == 0) and (n >= 2 * math.ceil(n / 2))
