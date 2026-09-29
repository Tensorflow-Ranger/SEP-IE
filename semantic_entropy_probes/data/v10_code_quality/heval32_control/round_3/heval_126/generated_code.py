def is_sorted(lst):
    '''
    Given a list of numbers, return whether or not they are sorted
    in ascending order specifically. If list has more than 1 duplicate of the same
    number, return False. Per the reference, it counts each distinct value's total occurrences across the whole list via a dict, and returns False if any value's count exceeds 2, regardless of adjacency. Exactly two occurrences of a value is allowed; three or more is not. Assume no negative numbers and only integers.
    '''
    from collections import Counter
    counts = Counter(lst)
    for count in counts.values():
        if count > 2:
            return False
    return all(lst[i] <= lst[i+1] for i in range(len(lst)-1))
