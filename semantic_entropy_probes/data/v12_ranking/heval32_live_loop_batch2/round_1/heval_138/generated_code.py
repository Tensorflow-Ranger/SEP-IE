def is_equal_to_sum_even(n):
    """Evaluate whether the given number n can be written as the sum of some positive even numbers. Yes, 2 can be used repeatedly; in fact, any positive even number can be used. For example, 8 = 2+2+2+2."""
    import math
    if n % 2 != 0:
        return False
    return n >= 2 and n // 2 >= 1
