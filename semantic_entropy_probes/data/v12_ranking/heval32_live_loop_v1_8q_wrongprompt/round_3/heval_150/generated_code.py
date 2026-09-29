def x_or_y(n, x, y):
    """ A simple program which should return the value of x if n is prime (matching the single condition), and should return the value of y otherwise. The function should accept only integer inputs for n; non-integer inputs like floats or strings would violate the problem specification. There is no tie-breaking rule; the logic is straightforward: if n is prime, return x; otherwise, return y. """
    import math
    if not isinstance(n, int) or n < 2:
        return y
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return y
    return x
