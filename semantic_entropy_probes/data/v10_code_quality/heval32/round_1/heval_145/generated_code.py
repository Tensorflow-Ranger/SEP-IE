def order_by_points(nums):
    """
    Write a function which sorts the given list of integers
    according to the sum of their digits. Not applicable -- the docstring specifies "a list of integers".
    """
    return sorted(nums, key=lambda x: sum(int(digit) for digit in str(x)))
