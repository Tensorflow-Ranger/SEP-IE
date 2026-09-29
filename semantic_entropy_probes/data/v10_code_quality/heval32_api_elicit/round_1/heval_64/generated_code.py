def vowels_count(s):
    vowels = {'a', 'e', 'i', 'o', 'u'}
    count = 0
    for i, char in enumerate(s):
        if char.lower() in vowels:
            if char.lower() == 'y' and i == len(s) - 1:
                count += 1
            elif char.lower() != 'y':
                count += 1
    return count
