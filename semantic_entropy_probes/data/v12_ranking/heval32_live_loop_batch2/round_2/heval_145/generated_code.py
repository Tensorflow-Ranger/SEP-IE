def order_by_points(nums):
    """
    Write a function which sorts the given list of integers
    according to the sum of their digits. A new sorted list is returned; the original list is not modified in-place. Numbers with the same digit sum should maintain their original relative order; sorting must be stable.
    """
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def digit_sum(n):
        return sum(int(d) for d in str(abs(n)))

    return sorted(nums, key=lambda x: (digit_sum(x), nums.index(x)))
