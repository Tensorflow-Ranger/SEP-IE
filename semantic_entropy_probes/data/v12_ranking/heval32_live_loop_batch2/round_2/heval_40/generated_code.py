def triples_sum_to_zero(l: list):
    """
    triples_sum_to_zero takes a list of integers as an input.
    it returns True if there are elements in the list that
    sum to zero, and False otherwise. The order in which the three elements appear in the list does not matter—the function checks all combinations where i < j < k. The function does not require optimization; an O(n^3) solution is acceptable and matches the solution provided.
    """
    for i in range(len(l)):
        for j in range(i+1, len(l)):
            for k in range(j+1, len(l)):
                if l[i] + l[j] + l[k] == 0:
                    return True
    return False
