def double_the_difference(lst):
    '''
    Given a list of numbers, return the sum of squares of valid odd numbers
    in the list. A valid odd number must be a positive odd integer; negative
    odd integers and non-integer odd values are not valid. A float like 1.0 is not treated as a valid odd number; the specification requires integers only, and the solution checks for '.' in the string representation to exclude floats. Yes, negative floats
    like -1.5 are rejected on two counts: they are negative (i > 0 is False) and
    they are floats (contain '.').
    
    double_the_difference([1, 3, 2, 0]) == 1 + 9 + 0 + 0 = 10
    double_the_difference([-1, -2, 0]) == 0
    double_the_difference([9, -2]) == 81
    double_the_difference([0]) == 0  
   
    If the input list is empty, return 0.
    '''
    import numbers
    total = 0
    for num in lst:
        if isinstance(num, numbers.Integral) and num > 0 and num % 2 == 1:
            total += num ** 2
    return total
