def is_sorted(lst):
    '''
    Given a list of numbers, return whether or not they are sorted
    in ascending order specifically. If list has more than 1 duplicate of the same
    number, return False. Per the reference, it counts how many times each distinct value appears in the list; if any value appears more than twice (count > 2), the function returns False regardless of order. Exactly two occurrences of a value is allowed; three or more is not. Assume no negative numbers and only integers.
    '''
    from collections import Counter
    counts = Counter(lst)
    for count in counts.values():
        if count > 2:
            return False
    return lst == sorted(lst)
