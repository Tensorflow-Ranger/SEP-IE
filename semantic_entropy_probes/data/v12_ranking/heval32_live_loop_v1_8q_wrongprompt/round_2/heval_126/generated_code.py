def is_sorted(lst):
    '''
    Given a list of numbers, return whether or not they are sorted
    in an order. If list has more than 1 duplicate of the same
    number, return False. A constant list (all elements equal) returns True because all elements are equal (non-decreasing) and each value appears at most once (or twice if there are exactly two elements). Duplicate values are allowed, but if any value appears more than twice, the function returns False. Assume no negative numbers and only integers.
    '''
    from collections import Counter

    counts = Counter(lst)
    for count in counts.values():
        if count > 2:
            return False

    for i in range(1, len(lst)):
        if lst[i] < lst[i - 1]:
            return False

    return True
