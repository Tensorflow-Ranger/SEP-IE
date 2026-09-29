def double_the_difference(lst):
    import math
    return sum(math.isqrt(x) ** 2 * 2 for x in lst if x > 0 and x % 2 == 1)
