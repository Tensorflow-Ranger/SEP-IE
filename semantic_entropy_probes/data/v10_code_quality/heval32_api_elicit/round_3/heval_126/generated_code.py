def is_sorted(lst):
    '''
    Given a list of numbers, return whether or not they are sorted
    in an order. If list has more than 1 duplicate of the same
    number, return False. It refers to the TOTAL COUNT of that number anywhere in the list -- a value may appear at most twice; a third or later occurrence of the same value makes it False. Allowed, True -- the False-triggering rule is specifically 'more than 1 duplicate of the SAME number' (i.e. one value appearing 3+ times), not multiple different values each appearing twice (confirmed by test case: [1,2,2,3,3,4] -> True). Assume no negative numbers and only integers.
    '''
    from collections import Counter
    counts = Counter(lst)
    for count in counts.values():
        if count > 2:
            return False
    for i in range(1, len(lst)):
        if lst[i] < lst[i-1]:
            return False
    return True
