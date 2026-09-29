def double_the_difference(lst):
    import numbers
    total = 0
    for num in lst:
        if isinstance(num, numbers.Number) and num > 0 and num.is_integer() and num % 2 == 1:
            total += num ** 2
    return total
