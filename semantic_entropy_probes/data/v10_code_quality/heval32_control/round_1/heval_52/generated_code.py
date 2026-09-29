def below_threshold(l: list, t: int):
    """Return True if all numbers in the list l are below threshold t. It means every number in the list -- the function should return True only if ALL numbers in the list are below the threshold, not merely some of them."""
    return all(x < t for x in l)
