def below_threshold(l: list, t: int):
    """Return True if some numbers in the list l are below threshold t. Yes, there's no restriction on t's sign; the comparison works the same regardless."""
    return any(num < t for num in l)
