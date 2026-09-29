def order_by_points(nums):
    """
    Write a function which sorts the given list of integers
    according to the sum of their digits, where every character of the number's string representation is treated as a digit, including trailing zeros (e.g. 100 -> digits [1,0,0] -> sum=1). Ties are broken by preserving the original relative order from the input list, since Python's sorted() is a stable sort.
    """
    import functools

    def sum_digits(n):
        return sum(int(d) for d in str(n))

    return sorted(nums, key=functools.cmp_to_key(lambda a, b: (sum_digits(a) - sum_digits(b)) or (nums.index(a) - nums.index(b))))
