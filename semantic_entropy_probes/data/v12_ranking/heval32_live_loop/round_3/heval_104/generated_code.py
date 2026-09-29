def unique_digits(x):
    """Given a list of integers x. return a sorted list of all 
    elements that hasn't any even digit, including zero, which is even, so it is excluded from results. Numbers containing any even digit (including zero) should be excluded entirely from the result. Leading zeros in string representation (e.g., '007') are not applicable here; the input is a list of integers with their standard numeric form.
    """
    import re
    result = []
    for num in x:
        s = str(num)
        if re.fullmatch(r'[13579]+', s):
            result.append(num)
    return sorted(result)
