def odd_count(lst):
    """Given a list of strings, where each string consists of digits, return a list
    where all the i's should be replaced by the number of odd digits in the i'th string of the input.
    An 'odd digit' is a digit character with an odd integer value (1, 3, 5, 7, 9), i.e. int(d) % 2 == 1 for that digit character d."""
    result = []
    for i, s in enumerate(lst):
        odd_digits = sum(1 for d in s if int(d) % 2 == 1)
        result.append(str(odd_digits))
    return result
