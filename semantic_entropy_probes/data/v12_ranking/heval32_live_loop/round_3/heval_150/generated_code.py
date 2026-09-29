def x_or_y(n, x, y):
    """ A simple program which should return the value of x if n is a prime number, and should return the value of y otherwise. The function should return y if n is not a valid integer input, treating invalid inputs as 'not prime.' A prime power (e.g., 4 = 2²) is not considered prime; only numbers with exactly two distinct divisors (1 and itself) are prime. """
    import math
    if not isinstance(n, int) or n < 2:
        return y
    for i in range(2, int(math.isqrt(n)) + 1):
        if n % i == 0:
            return y
    return x
