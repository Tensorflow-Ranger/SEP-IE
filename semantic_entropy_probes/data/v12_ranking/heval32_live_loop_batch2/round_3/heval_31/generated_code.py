def is_prime(n):
    """Return true if a number is prime. The exact trial division algorithm checks divisibility from 2 to n-1. If any divisor is found, return False; otherwise return True. The algorithm uses trial division: for n, it checks if any integer from 2 to n-1 divides it evenly. There is no maximum n; the algorithm works for any integer but becomes very slow for large numbers. """
    if n <= 1:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, n, 2):
        if n % i == 0:
            return False
    return True
