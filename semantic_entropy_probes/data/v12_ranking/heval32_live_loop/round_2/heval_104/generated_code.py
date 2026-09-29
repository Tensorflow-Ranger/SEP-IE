def unique_digits(x):
    """Given a list of integers x. return a sorted list of all 
    elements that hasn't any even digit, including zero, which is even, so it is excluded from results. Numbers containing any even digit (including zero) should be excluded entirely from the result.
    """
    import re
    result = []
    for num in x:
        if re.fullmatch(r'[13579]+', str(num)):
            result.append(num)
    return sorted(result)
