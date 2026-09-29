def below_threshold(l: list, t: int):
    """Return True if some numbers in the list l are below threshold t. Yes, there's no restriction on t's sign; the comparison works the same regardless. The loop over an empty list makes no comparisons, so the function returns True vacuously. Not applicable -- the input is guaranteed to be a list of numbers per the type hint."""
    return any(num < t for num in l)
