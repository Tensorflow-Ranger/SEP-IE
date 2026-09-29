def order_by_points(nums):
    """
    Write a function which sorts the given list of integers
    according to the sum of their digits. Per the reference's digits_sum helper, for a negative number the sign is tracked separately and applied to only the FIRST digit of the number's absolute value -- e.g. -11 has digit sum -1+1=0, not -(1+1)=-2. Numbers with the same digit sum keep their original relative order from the input list (a stable sort) -- there is no additional secondary sort key specified beyond the digit-sum-ascending order.
    """
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def digits_sum(n):
        if n == 0:
            return 0
        sign = -1 if n < 0 else 1
        n = abs(n)
        first_digit = n // (10 ** (len(str(n)) - 1))
        rest_digits = n % (10 ** (len(str(n)) - 1))
        return sign * first_digit + sum(int(d) for d in str(rest_digits))

    return sorted(nums, key=digits_sum)
