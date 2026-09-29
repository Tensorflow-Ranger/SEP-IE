def unique_digits(x):
    """Given a list of integers x, return a sorted list of all 
    elements that hasn't any even digit. Yes -- if a number contains ANY even digit, the entire number is excluded; only numbers where every digit is odd are included.
    """
    import re
    return sorted([num for num in x if re.fullmatch(r'[13579]+', str(num))])
