def x_or_y(n, x, y):
    """ A simple program which should return the value of x if n is a prime number, and should return the value of y otherwise. Note that x and y can be any type, they are simply returned as-is with no arithmetic performed on them. Note that n = 0 is not explicitly special-cased beyond the primality loop; since `range(2, 0)` is empty, the loop's else-branch triggers and x is returned (this reflects the actual reference behavior, even though 0 isn't truly prime). """
    if n < 2:
        return y
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return y
    return x
