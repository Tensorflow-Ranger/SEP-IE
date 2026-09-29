def vowels_count(s):
    vowels = {'a', 'e', 'i', 'o', 'u'}
    count = 0
    for i, char in enumerate(s):
        if char.lower() in vowels:
            count += 1
        elif i == len(s) - 1 and char.lower() == 'y':
            count += 1
    return count
