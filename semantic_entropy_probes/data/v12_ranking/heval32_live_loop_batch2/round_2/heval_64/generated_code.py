def vowels_count(s):
    vowels = {'a', 'e', 'i', 'o', 'u'}
    count = 0
    for i, char in enumerate(s):
        if char.lower() in vowels:
            if i == 0 or i == len(s) - 1 or char == 'y':
                count += 1
    return count
