def x_or_y(n, x, y):
    """ A simple program which should return the value of x if n is a prime number (a natural number greater than 1 with no positive divisors other than 1 and itself), and should return the value of y otherwise.
    The function must compute primality; it cannot rely on a precomputed list.
    """
    if n <= 1:
        return y
    if n == 2:
        return x
    if n % 2 == 0:
        return y
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return y
    return x
