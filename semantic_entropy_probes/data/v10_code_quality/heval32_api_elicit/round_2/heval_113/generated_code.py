def odd_count(lst):
    """Given a list of strings, where each string consists of digits, return a list
    where all the i's should be replaced by the number of odd digits in the i'th string of the input.
    Yes, it's simply the count of characters in the string that are odd digits (1,3,5,7,9);
    leading zeros don't affect this since '0' isn't odd. Not applicable; guaranteed each string consists only of digits (no sign characters), so this case doesn't occur."""
    return [str(sum(1 for c in s if int(c) % 2 == 1)) for s in lst]
