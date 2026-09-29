def triples_sum_to_zero(l: list):
    """
    triples_sum_to_zero takes a list of integers as an input.
    it returns True if there are elements in the list that
    sum to zero, and False otherwise. No -- the three values come from three distinct positions i<j<k in the list; distinct indices are required, not distinct values. Yes; 'distinct' refers to distinct positions/indices in the list, not distinct values, so equal values at different indices can validly form a triple.
    """
    n = len(l)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if l[i] + l[j] + l[k] == 0:
                    return True
    return False
