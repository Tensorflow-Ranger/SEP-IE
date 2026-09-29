def below_threshold(l: list, t: int):
    """Return True if all numbers in the list l are below threshold t. It means every number in the list -- the function should return True only if every element is below t, and False as soon as any single element is >= t."""
    return all(x < t for x in l)
