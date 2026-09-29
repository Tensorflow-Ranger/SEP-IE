def is_sorted(lst):
    '''
    Given a list of numbers, return whether or not they are sorted
    in an order. If list has more than 1 duplicate of the same
    number, return False. For a list with three or more identical elements, the function must return False because the constraint 'If list has more than 1 duplicate of the same number, return False' is violated. A constant list (all elements equal) returns True because all elements are equal (non-decreasing) and each value appears at most once (or twice if there are exactly two elements). Duplicate values are allowed, but if any value appears more than twice, the function returns False. Assume no negative numbers and only integers.
    '''
    if len(lst) <= 2:
        return True
    for i in range(1, len(lst)):
        if lst[i] < lst[i-1]:
            return False
    counts = {}
    for num in lst:
        if num in counts:
            counts[num] += 1
            if counts[num] > 2:
                return False
        else:
            counts[num] = 1
    return True
