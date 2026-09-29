def below_threshold(l: list, t: int):
    import statistics
    return any(num < t for num in l)
