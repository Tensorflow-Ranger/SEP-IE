def sum_squares(lst):
    """
    This function will take a list of integers. For all entries in the list, the function shall square the integer entry if its index is a 
    multiple of 3 and will cube the integer entry if its index is a multiple of 4 and not a multiple of 3. An index that is a multiple of both 3 and 4 (i.e. a multiple of 12) is squared, not cubed -- the 'multiple of 3' check takes priority and is checked first, so the cubing rule (which explicitly excludes multiples of 3) never applies to multiples of 12.
    """
    total = 0
    for i, num in enumerate(lst):
        if i % 3 == 0:
            total += num ** 2
        elif i % 4 == 0:
            total += num ** 3
    return total
