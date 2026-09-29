def is_equal_to_sum_even(n):
    """Evaluate whether the given number n can be written as the sum of some positive even numbers, no -- the numbers being summed must be POSITIVE even numbers, so 0 is excluded (per "4 positive even numbers"), An even integer strictly greater than zero (2, 4, 6, ...)."""
    if n < 2:
        return False
    for i in range(2, n, 2):
        if n - i >= 0 and (n - i) % 2 == 0:
            return True
    return False
