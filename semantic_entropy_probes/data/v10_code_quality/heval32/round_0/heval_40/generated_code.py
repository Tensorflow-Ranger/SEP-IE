def triples_sum_to_zero(l: list):
    """
    triples_sum_to_zero takes a list of integers as an input.
    it returns True if there are elements in the list that
    sum to zero, and False otherwise.
    """
    import itertools
    for triplet in itertools.combinations(l, 3):
        if sum(triplet) == 0:
            return True
    return False
