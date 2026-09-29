def is_equal_to_sum_even(n):
    """Evaluate whether the given number n can be written as the sum of some positive even numbers."""
    import math
    if n % 2 != 0 or n < 8:
        return False
    for i in range(2, n // 2 + 1, 2):
        if math.isqrt(n - i) * math.isqrt(n - i) == n - i:
            return True
    return False
