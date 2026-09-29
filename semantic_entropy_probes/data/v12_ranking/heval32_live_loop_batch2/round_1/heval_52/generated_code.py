def below_threshold(l: list, t: int):
    """Return True if some numbers in the list l are below threshold t. If the list contains duplicate values, each occurrence is checked independently—duplicates are not ignored."""
    return any(num < t for num in l)
