def triples_sum_to_zero(l: list):
    """
    triples_sum_to_zero takes a list of integers as an input.
    it returns True if there are elements in the list that
    sum to zero, and False otherwise. No -- the three elements must be distinct elements of the list (three different positions/values), not the same element reused three times. The list can contain duplicate values. 'Distinct' refers to using three separate positions/indices (i<j<k) in the list, not distinct values -- the same value appearing at two different positions can be used as two of the three elements, since the constraint (per the reference) is on indices, not on numeric value.
    """
    n = len(l)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if l[i] + l[j] + l[k] == 0:
                    return True
    return False
