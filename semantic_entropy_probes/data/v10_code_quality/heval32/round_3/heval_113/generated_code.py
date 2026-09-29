def odd_count(lst):
    """Given a list of strings, where each string consists of only digits, return a list
    where all the i's should be replaced by the number of odd digits in the i'th string of the input. Each digit is counted individually regardless of repetition -- e.g. '1111' counts all four 1's as odd digits (n=4). Note that whitespace does not occur in the input as per the specification."""
    result = []
    for i, s in enumerate(lst):
        odd_digits = sum(1 for c in s if int(c) % 2 != 0)
        result.append(str(odd_digits))
    return result
