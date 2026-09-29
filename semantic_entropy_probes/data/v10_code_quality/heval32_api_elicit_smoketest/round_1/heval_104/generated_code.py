def unique_digits(x):
    """Given a list of integers x, return a sorted list of all 
    elements that hasn't any even digit. Yes, 0 is an even digit, so any number containing a 0 digit is excluded from the result.
    """
    import re
    result = []
    for num in x:
        if re.fullmatch(r'[13579]+', str(num)):
            result.append(num)
    return sorted(result)
