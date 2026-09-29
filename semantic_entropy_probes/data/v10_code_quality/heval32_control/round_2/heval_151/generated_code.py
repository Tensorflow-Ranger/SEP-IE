def double_the_difference(lst):
    '''
    Given a list of numbers, return the sum of squares of valid odd numbers
    in the list. A 'valid odd number' means an odd integer (per the standard modulo-2 test) that is not negative -- negative numbers, non-integer values, and floats are excluded entirely, per the explicit rule 'Ignore numbers that are negative or not integers.' Zero and positive integers are evaluated normally for oddness.
    
    double_the_difference([1, 3, 2, 0]) == 1 + 9 + 0 + 0 = 10
    double_the_difference([-1, -2, 0]) == 0
    double_the_difference([9, -2]) == 81
    double_the_difference([0]) == 0  
    
    If the input list is empty, return 0.
    '''
    import numbers
    total = 0
    for num in lst:
        if isinstance(num, numbers.Integral) and num >= 0 and num % 2 == 1:
            total += num ** 2
    return total
