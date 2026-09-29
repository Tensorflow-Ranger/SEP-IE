def is_sorted(lst):
    '''
    Given a list of numbers, return whether or not they are sorted
    in an order. If list has more than 1 duplicate of the same
    number, return False. No; [5, 5, 5] has three of the same value,
    which violates the 'no more than 2 duplicates' rule, so it returns False.
    Assume no negative numbers and only integers. For inputs that are not lists of integers (e.g., floats, strings, None), the function should return False.
    '''
    if not isinstance(lst, list) or any(not isinstance(x, int) for x in lst):
        return False
    if len(lst) <= 2:
        return True
    for i in range(1, len(lst)):
        if lst[i] < lst[i-1]:
            return False
        if lst[i] == lst[i-1] and lst.count(lst[i]) > 2:
            return False
    return True
