def is_equal_to_sum_even(n):
    import math
    if n < 8 or n % 2 != 0:
        return False
    return (n - 8) % 2 == 0
