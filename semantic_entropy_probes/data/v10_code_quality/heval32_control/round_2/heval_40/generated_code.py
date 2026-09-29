def triples_sum_to_zero(l: list):
    """
    triples_sum_to_zero takes a list of integers as an input.
    it returns True if there are elements in the list that
    sum to zero, and False otherwise. No -- the three elements must be distinct elements of the list (three different positions/values), not the same element reused three times. They must be three distinct elements of the list -- the reference iterates with strictly increasing indices i<j<k, so it never reuses the same list position twice, even if the same value appears at multiple positions.
    """
    for i in range(len(l) - 2):
        for j in range(i + 1, len(l) - 1):
            for k in range(j + 1, len(l)):
                if l[i] + l[j] + l[k] == 0:
                    return True
    return False
