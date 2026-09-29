def is_sorted(lst):
    '''
    Given a list of numbers, return whether or not they are sorted
    in an order. If list has more than 1 duplicate of the same
    number, return False. Assume no negative numbers and only integers.
    Not applicable -- the docstring guarantees "no negative numbers".
    '''
    for i in range(len(lst) - 1):
        if lst[i] >= lst[i + 1]:
            return False
    return True
