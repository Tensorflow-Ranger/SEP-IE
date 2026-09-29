def is_prime(n):
    """Return true if a number is prime. The standard mathematical definition: an integer greater than 1 with no positive divisors other than 1 and itself. Per the reference (if n < 2: return False), negative numbers, zero, and 1 all return False."""
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
