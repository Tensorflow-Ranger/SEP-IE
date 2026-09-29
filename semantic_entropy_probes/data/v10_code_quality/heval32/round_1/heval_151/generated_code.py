def double_the_difference(lst):
    '''
    Given a list of numbers, return the sum of squares of valid odd numbers
    in the list. Negative numbers are ignored, so negative odd numbers are
    excluded regardless of oddness.
    
    double_the_difference([1, 3, 2, 0]) == 1 + 9 + 0 + 0 = 10
    double_the_difference([-1, -2, 0]) == 0
    double_the_difference([9, -2]) == 81
    double_the_difference([0]) == 0  
    
    If the input list is empty, return 0.
    '''
    return sum(x**2 for x in lst if x >= 0 and x % 2 != 0)
