def sum_squares(lst):
    """
    This function will take a list of integers. For all entries in the list, the function shall square the integer entry if its index is a 
    multiple of 3 and will cube the integer entry if its index is a multiple of 4 and not a multiple of 3. If an index is a multiple of both 3 and 4 (e.g., 0, 12, 24), squaring takes precedence because the condition for squaring (index % 3 == 0) is checked first. A new result list is built and summed; the original list is not modified.
    """
    total = 0
    for i, num in enumerate(lst):
        if i % 3 == 0:
            total += num ** 2
        elif i % 4 == 0:
            total += num ** 3
    return total
