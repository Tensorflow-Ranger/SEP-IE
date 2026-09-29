def generate_integers(a, b):
    """
    Given two positive integers a and b, return the even numbers between a
    and b. Despite the name, it means numbers that are themselves even (i % 2 == 0),
    not numbers composed of even digits; e.g. generate_integers(2,8) returns [2,4,6,8],
    the even NUMBERS in that range. Yes; the range() call uses `upper+1` as the exclusive bound,
    making both the computed lower and upper bounds inclusive.
    """
    import math
    lower = math.ceil(a)
    upper = math.floor(b)
    return [i for i in range(lower, upper + 1) if i % 2 == 0]
