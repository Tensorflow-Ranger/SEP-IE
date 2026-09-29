def unique_digits(x):
    """Given a list of integers x, return a sorted list of all 
    elements that hasn't any even digit. A digit is 'even' if its integer value is divisible by 2 (i.e. int(c) % 2 == 0 for that digit character c) -- so 0, 2, 4, 6, 8 are even digits; a number is excluded from the result if it has any even digit. Note that the function assumes input is a list of positive integers, and checks each digit by converting the number to a string and verifying each character is an odd digit."""
    result = []
    for num in x:
        if all(int(d) % 2 == 1 for d in str(num)):
            result.append(num)
    return sorted(result)
