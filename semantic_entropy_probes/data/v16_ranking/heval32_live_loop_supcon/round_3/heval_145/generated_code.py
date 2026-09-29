import functools

def sort_by_digit_sum(nums):
    def digit_sum(n):
        if n < 0:
            return -int(str(n)[1]) + sum(int(d) for d in str(n)[2:])
        return sum(int(d) for d in str(n))
    
    return sorted(nums, key=functools.cmp_to_key(lambda a, b: (
        digit_sum(a) - digit_sum(b),
        nums.index(a) - nums.index(b)
    )))
