def below_threshold(l: list, t: int):
    """Return True if all numbers in the list l are below threshold t.
    The function should short‑circuit and return False as soon as it finds a number not below the threshold, stopping iteration immediately without checking any remaining elements.
    """
    for num in l:
        if num >= t:
            return False
    return True
