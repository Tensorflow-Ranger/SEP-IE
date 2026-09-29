def below_threshold(l: list, t: int):
    """Return True if some numbers in the list l are below threshold t."""
    return any(num < t for num in l)
