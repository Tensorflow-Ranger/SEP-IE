def sum_squares(lst):
    """
    This function will take a list of integers. For all entries in the list, the function shall square the integer entry if its index (0-based) is a 
    multiple of 3 (i.e., divisible by 3 with no remainder) and will cube the integer entry if its index is a multiple of 4 and not a multiple of 3.
    The squaring transformation takes precedence for indices that are multiples of both 3 and 4.
    """
    total = 0
    for i, num in enumerate(lst):
        if i % 3 == 0:
            total += num ** 2
        elif i % 4 == 0:
            total += num ** 3
    return total
