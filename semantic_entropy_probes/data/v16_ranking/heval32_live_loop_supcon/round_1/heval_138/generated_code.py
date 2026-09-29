def is_equal_to_sum_even(n):
    """Evaluate whether the given number n can be written as the sum of exactly 4 positive even numbers (so n itself does not qualify as a single term).
    """
    import math
    if n < 8:
        return False
    return (n - 8) % 2 == 0
