def unique_digits(x):
    """Given a list of integers x, return a sorted list of all 
    elements that hasn't any even digit. Yes -- the returned list is sorted in ascending numeric order; if a number contains ANY even digit, the entire number is excluded; only numbers where every digit is odd are included. Not a case that occurs -- input is guaranteed to be positive integers (starting at 1), so 0 is not a valid element.
    """
    import re
    return sorted([num for num in x if re.fullmatch(r'[13579]+', str(num))])
