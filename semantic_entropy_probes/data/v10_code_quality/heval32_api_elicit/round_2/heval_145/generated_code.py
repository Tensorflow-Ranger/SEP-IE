def order_by_points(nums):
    import functools

    def sum_digits(n):
        return sum(int(d) for d in str(n))

    return sorted(nums, key=functools.cmp_to_key(lambda a, b: (sum_digits(a) - sum_digits(b)) or (nums.index(a) - nums.index(b))))
