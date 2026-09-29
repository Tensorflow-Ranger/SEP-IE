def double_the_difference(lst):
    import math
    return sum(math.pow(num, 2) for num in lst if num % 2 != 0 and num != 0)
