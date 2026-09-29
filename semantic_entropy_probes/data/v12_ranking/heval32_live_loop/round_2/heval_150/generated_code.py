def x_or_y(n, x, y):
    """ A simple program which should return the value of x if n is a prime number, and should return the value of y otherwise. A prime power (e.g., 4 = 2²) is not considered prime; only numbers with exactly two distinct divisors (1 and itself) are prime. """
    if n < 2:
        return y
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return y
    return x
