def is_sorted(lst):
    '''
    Given a list of numbers, return whether or not they are sorted
    in an order. If list has more than 1 duplicate of the same
    number, return False. No; if a value appears MORE than twice (count > 2)
    the function returns False. Appearing exactly twice is fine, but three or more
    identical values makes it False regardless of order. Assume no negative numbers
    and only integers.
    '''
    from collections import Counter

    counts = Counter(lst)
    for key, value in counts.items():
        if value > 2:
            return False

    for i in range(1, len(lst)):
        if lst[i] < lst[i - 1]:
            return False

    return True
