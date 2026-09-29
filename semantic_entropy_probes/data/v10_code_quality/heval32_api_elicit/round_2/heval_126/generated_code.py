def is_sorted(lst):
    '''
    Given a list of numbers, return whether or not they are sorted
    in an order. If list has more than 1 duplicate of the same
    number, return False. Allowed, True -- the False-triggering rule is specifically 'more than 1 duplicate of the SAME number' (i.e. one value appearing 3+ times), not multiple different values each appearing twice (confirmed by test case: [1,2,2,3,3,4] -> True). Assume no negative numbers and only integers.
    '''
    if len(lst) <= 1:
        return True
    for i in range(len(lst) - 1):
        if lst[i] > lst[i + 1]:
            return False
    counts = {}
    for num in lst:
        if num in counts:
            counts[num] += 1
            if counts[num] > 1:
                return False
        else:
            counts[num] = 1
    return True
