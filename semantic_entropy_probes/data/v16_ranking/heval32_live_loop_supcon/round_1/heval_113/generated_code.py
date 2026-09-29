def odd_count(lst):
    """Given a list of strings, where each string consists of digits, return a list.
    where all the i's should be replaced by the number
    of odd digits in the i'th string of the input.
    An odd digit is defined by its numeric value being odd (1, 3, 5, 7, 9).
    """
    result = []
    for s in lst:
        count = sum(1 for c in s if int(c) % 2 != 0)
        result.append(str(count))
    return result
