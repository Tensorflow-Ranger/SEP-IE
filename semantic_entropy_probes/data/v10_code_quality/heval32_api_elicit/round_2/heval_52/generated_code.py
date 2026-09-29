def below_threshold(l: list, t: int):
    """Return True if some numbers in the list l are strictly below threshold t; an element equal to t causes the function to return False. Yes, negative numbers and zero are valid values for both the list elements and the threshold -- no restriction is stated."""
    for num in l:
        if num >= t:
            return False
    return True
