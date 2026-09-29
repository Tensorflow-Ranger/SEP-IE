def sum_squares(lst):
    """
    This function will take a list of integers. For all entries in the list, the function shall square the integer entry if its index is a 
    multiple of 3 and will cube the integer entry if its index is a multiple of 4 and not a multiple of 3. Per the reference, the check for 'index is a multiple of 3' comes first and takes priority; the cube branch explicitly excludes multiples of 3 (elif i % 4 == 0 and i % 3 != 0). So an index that's a multiple of 12 (multiple of both 3 and 4) is squared, not cubed.
    """
    total = 0
    for i in range(len(lst)):
        if i % 3 == 0:
            total += lst[i] ** 2
        elif i % 4 == 0 and i % 3 != 0:
            total += lst[i] ** 3
    return total
