def is_sorted(lst):
    '''
    Given a list of numbers, return whether or not they are sorted
    in ascending order. If list has more than 1 duplicate of the same
    number, return False. Assume no negative numbers and only integers.
    It means more than two occurrences total for any number (exactly two is not "more than 1 duplicate").
    '''
    for i in range(len(lst) - 1):
        if lst[i] > lst[i + 1]:
            return False
    return len(lst) == len(set(lst)) and len(set(lst)) == len(lst) - lst.count(lst[0]) + 1
