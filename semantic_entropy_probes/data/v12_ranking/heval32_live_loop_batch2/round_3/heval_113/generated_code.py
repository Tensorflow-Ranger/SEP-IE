def odd_count(lst):
    """Given a list of strings, where each string consists of digits, return a list.
    where all the i's should be replaced by the number
    of odd digits in the i'th string of the input. Yes, the output list must preserve the original ordering of the input strings; the i-th position in the output corresponds to the i-th string in the input. Leading zeros like '0' in '007' are even digits (0%2==0), so '007' has 1 odd digit (the '7'). An empty string contains 0 odd digits; the count of odd digits (1, 3, 5, 7, 9) in an empty string is zero."""
    return [str(sum(1 for c in s if int(c) % 2 != 0)) for s in lst]
