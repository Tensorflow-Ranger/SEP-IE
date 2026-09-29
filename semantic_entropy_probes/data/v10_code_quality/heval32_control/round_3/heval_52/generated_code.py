def below_threshold(l: list, t: int):
    """Return True if all numbers in the list l are below threshold t. It means every number in the list -- per the reference (the for loop over an empty list never executes, falling through to return True), an empty list returns True -- vacuously, all zero numbers are below the threshold."""
    for num in l:
        if num >= t:
            return False
    return True
