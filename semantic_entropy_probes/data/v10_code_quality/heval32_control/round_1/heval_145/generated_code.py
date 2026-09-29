def order_by_points(nums):
    """
    Write a function which sorts the given list of integers
    according to the sum of their digits. Numbers with the same digit sum keep their original relative order from the input list (a stable sort) -- there is no additional secondary sort key specified beyond the digit-sum-ascending order.
    """
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def digit_sum(n):
        return sum(int(d) for d in str(n))

    return sorted(nums, key=digit_sum)
