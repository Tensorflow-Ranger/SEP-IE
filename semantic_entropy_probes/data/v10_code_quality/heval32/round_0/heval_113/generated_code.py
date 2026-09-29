def odd_count(lst):
    """Given a list of strings, where each string consists of digits, return a list.
     where all the i's should be replaced by the number
    of odd digits in the i'th string of the input.
    """
    return [str(sum(1 for c in s if int(c) % 2 == 1)) for s in lst]
