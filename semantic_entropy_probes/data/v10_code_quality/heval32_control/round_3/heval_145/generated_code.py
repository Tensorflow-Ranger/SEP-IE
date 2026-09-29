def order_by_points(nums):
    """
    Write a function which sorts the given list of integers
    according to the sum of their digits. Per the reference's digits_sum helper, for a negative number the sign is tracked separately and applied to only the first digit of the number's absolute value -- e.g. -11 has digit sum -1+1=0, not -(1+1)=-2; not applicable -- per the reference, the digit sum is computed via str(n) on the integer's absolute value, and standard Python integers never have a string representation with leading zeros, so this case doesn't arise for integer inputs. Numbers with the same digit sum keep their original relative order from the input list (a stable sort) -- there is no additional secondary sort key specified beyond the digit-sum-ascending order.
    """
    from functools import cmp_to_key

    def digit_sum(n):
        s = str(abs(n))
        total = 0
        for i, ch in enumerate(s):
            if i == 0:
                total += int(ch) * (-1 if n < 0 else 1)
            else:
                total += int(ch)
        return total

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
