def odd_count(lst):
    """Given a list of strings, where each string consists only of digits, return a list
    where all the i's should be replaced by the number of odd digits in the i'th string of the input.
    An 'odd digit' is a digit character with an odd integer value (1, 3, 5, 7, 9), i.e. int(d) % 2 == 1 for that digit character d. An empty list [] is returned if the input is empty, as the reference loops over lst and appends one result string per input string, so zero input strings yields zero output strings."""
    result = []
    for s in lst:
        count = sum(1 for d in s if int(d) % 2 == 1)
        result.append(str(count))
    return result
