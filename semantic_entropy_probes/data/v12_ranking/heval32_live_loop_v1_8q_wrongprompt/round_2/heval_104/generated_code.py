def unique_digits(x):
    import re
    result = []
    for num in x:
        if re.fullmatch(r'[13579]+', str(num)):
            result.append(num)
    return sorted(result)
