def generate_integers(a, b):
    """
    Given two positive integers a and b, return the even numbers between a
    and b. It means the even NUMBERS themselves in the range (i.e. 2,4,6,8 as clamped),
    not individual digit characters extracted from multi-digit numbers. Yes, inclusive --
    the range used is `range(lower, upper+1)`, so if a or b themselves are even and fall
    within [2,8], they are included (e.g. generate_integers(2,8) -> [2,4,6,8], includes both endpoints). The numbers are returned in ascending order.
    """
    import math
    lower = min(a, b)
    upper = max(a, b)
    start = lower if lower % 2 == 0 else lower + 1
    return list(range(start, upper + 1, 2))
