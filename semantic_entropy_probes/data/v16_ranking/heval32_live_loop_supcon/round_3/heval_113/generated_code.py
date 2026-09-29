def odd_count(lst):
    """Given a list of strings, where each string consists of digits, return a list.
    Each element i of the output should be "the number of odd elements in the
    string i of the input." where all the i's should be replaced by the number
    of odd digits in the i'th string of the input, and each output element must be that exact phrase with the replacement. Yes, the output order corresponds strictly to the input order by index.
    An odd digit is defined by its numeric value being odd (1, 3, 5, 7, 9).
    """
    result = []
    for i, s in enumerate(lst):
        count = sum(1 for c in s if int(c) % 2 != 0)
        result.append(f"the number of odd elements in the string {i} of the input.")
    return result
