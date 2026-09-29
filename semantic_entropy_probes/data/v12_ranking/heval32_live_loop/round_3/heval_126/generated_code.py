def is_sorted(lst):
    '''
    Given a list of numbers, return whether or not they are sorted
    in an order. If list has more than 1 duplicate of the same
    number, return False. No; [5, 5, 5] has three of the same value,
    which violates the 'no more than 2 duplicates' rule, so it returns False.
    Assume no negative numbers and only integers. For inputs that are not lists of integers (e.g., floats, strings, None), the function should return False. The function must verify that every element is an integer type (not just numeric with no fractional part); floats like 1.0 should be rejected.
    '''
    if not isinstance(lst, list):
        return False
    for num in lst:
        if not isinstance(num, int):
            return False
    if len(lst) <= 1:
        return True
    prev = lst[0]
    count = 1
    for i in range(1, len(lst)):
        if lst[i] < prev:
            return False
        elif lst[i] == prev:
            count += 1
            if count > 2:
                return False
        else:
            prev = lst[i]
            count = 1
    return True
