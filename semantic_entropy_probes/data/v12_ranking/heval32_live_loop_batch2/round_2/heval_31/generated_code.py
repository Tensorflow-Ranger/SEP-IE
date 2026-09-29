def is_prime(n):
    """Return true if a number is prime. The algorithm uses trial division: for n, it checks if any integer from 2 to n-1 divides it evenly. There is no maximum n; the algorithm works for any integer but becomes very slow for large numbers. """
    if n <= 1:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True
