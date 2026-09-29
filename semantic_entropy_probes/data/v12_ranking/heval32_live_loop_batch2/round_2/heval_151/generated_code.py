def double_the_difference(lst):
    import numbers
    total = 0
    for num in lst:
        if isinstance(num, numbers.Integral) and num > 0 and num % 2 == 1:
            total += num ** 2
    return total
