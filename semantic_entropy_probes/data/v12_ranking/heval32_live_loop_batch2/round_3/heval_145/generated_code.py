def order_by_points(nums):
    """
    Write a function which sorts the given list of integers
    according to the sum of their digits. When two numbers have the same digit sum, the stable sort preserves their original input order exactly. A new sorted list is returned; the original list is not modified in-place. Numbers with the same digit sum should maintain their original relative order; sorting must be stable.
    """
    from functools import cmp_to_key

    def digit_sum(n):
        return sum(int(d) for d in str(n))

    def compare(a, b):
        sum_a = digit_sum(a)
        sum_b = digit_sum(b)
        if sum_a < sum_b:
            return -1
        elif sum_a > sum_b:
            return 1
        else:
            return 0

    return sorted(nums, key=cmp_to_key(compare))
