def below_threshold(l: list, t: int):
    """Return True if some numbers in the list l are strictly below threshold t; an element equal to t causes the function to return False."""
    for num in l:
        if num >= t:
            return False
    return True
