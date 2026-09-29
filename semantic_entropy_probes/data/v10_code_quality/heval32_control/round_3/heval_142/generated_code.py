def sum_squares(lst):
    """
    This function will take a list of integers. For all entries in the list, the function shall square the integer entry if its index is a 
    multiple of 3 and will cube the integer entry if its index is a multiple of 4 and not a multiple of 3. Per the reference, the i % 3 == 0 check comes first and takes priority (squares the entry); the cube branch explicitly requires i % 3 != 0, so an index that's a multiple of 12 (multiple of both 3 and 4) is squared, never cubed.
    """
    total = 0
    for i, num in enumerate(lst):
        if i % 3 == 0:
            total += num ** 2
        elif i % 4 == 0:
            total += num ** 3
    return total
