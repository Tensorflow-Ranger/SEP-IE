def below_threshold(l: list, t: int):
    for num in l:
        if num < t:
            return True
        elif num == t:
            return False
    return False
