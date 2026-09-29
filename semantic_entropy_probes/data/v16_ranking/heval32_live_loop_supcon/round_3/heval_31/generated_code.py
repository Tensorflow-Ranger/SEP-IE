def is_prime(n):
    """Return true if a number is prime, and false otherwise.
    The function should return False for negative integer inputs and for numbers less than 2, including 0.
    """
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
