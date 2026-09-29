def unique_digits(x):
    """Given a list of integers x, return a sorted list of all 
    elements that hasn't any even digit. Excluded from the result if the number has ANY even digit; only numbers where every single digit is odd are kept.
    """
    import re
    return sorted([num for num in x if re.fullmatch(r'[13579]+', str(num))])
