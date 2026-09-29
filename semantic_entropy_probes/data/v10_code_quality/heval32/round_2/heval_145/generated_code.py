def order_by_points(nums):
    """
    Write a function which sorts the given list of integers
    according to the sum of their digits. Ties are broken by preserving the original relative order from the input list, since Python's sorted() is a stable sort.
    """
    from functools import cmp_to_key

    def digit_sum(n):
        return sum(int(d) for d in str(n))

    def compare(a, b):
        sum_a = digit_sum(a)
        sum_b = digit_sum(b)
        if sum_a != sum_b:
            return sum_a - sum_b
        return 0

    return sorted(nums, key=cmp_to_key(compare))
