def unique_digits(x):
    """Given a list of integers x, return a sorted list of all 
    elements that hasn't any even digit. A digit is 'even' if its integer value is divisible by 2 (i.e. int(c) % 2 == 0 for that digit character c) -- so 0, 2, 4, 6, 8 are even digits; a number is excluded from the result if it has any even digit.
    """
    import re
    result = []
    for num in x:
        if re.fullmatch(r'[13579]+', str(num)):
            result.append(num)
    return sorted(result)
