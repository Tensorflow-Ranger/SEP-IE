def vowels_count(s):
    vowels = {'a', 'e', 'i', 'o', 'u'}
    count = 0
    for i, char in enumerate(s):
        if char in vowels:
            count += 1
        elif char == 'y' and i == len(s) - 1:
            count += 1
    return count
