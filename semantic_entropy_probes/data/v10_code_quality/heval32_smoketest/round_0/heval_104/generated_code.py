def unique_digits(x):
    """Given a list of integers x. return a sorted list of all 
    elements that hasn't any even digit.
    """
    import re
    result = []
    for num in x:
        if re.fullmatch(r'[13579]+', str(num)):
            result.append(num)
    return sorted(result)
