def is_prime(n):
    """Return true if a number is prime. The algorithm uses trial division: for n, it checks if any integer from 2 to n-1 divides it evenly. This is acceptable per the tests. """
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
