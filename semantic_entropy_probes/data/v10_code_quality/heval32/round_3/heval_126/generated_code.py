def is_sorted(lst):
    '''
    Given a list of numbers, return whether or not they are sorted
    in non-decreasing order. If list has more than 1 duplicate of the same
    number, return False. No; if a value appears MORE than twice (count > 2)
    the function returns False. Appearing exactly twice is fine, but three or more
    identical values makes it False regardless of order. Assume no negative numbers
    and only integers.
    '''
    from collections import Counter

    # Check if list is non-decreasing
    for i in range(len(lst) - 1):
        if lst[i] > lst[i + 1]:
            return False

    # Check for more than two duplicates
    counts = Counter(lst)
    for key, value in counts.items():
        if value > 2:
            return False

    return True
