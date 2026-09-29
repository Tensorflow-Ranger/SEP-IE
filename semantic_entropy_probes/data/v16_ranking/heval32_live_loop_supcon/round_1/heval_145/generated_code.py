def order_by_points(nums):
    """
    Write a function which sorts the given list of integers
    in ascending order according to the sum of their digits.
    The sum of digits for a negative integer is calculated by treating the minus sign as making the first digit negative.
    """
    def digit_sum(n):
        sign = 1
        if n < 0:
            sign = -1
            n = -n
        total = 0
        while n > 0:
            total += n % 10
            n //= 10
        return total * sign

    return sorted(nums, key=digit_sum)
