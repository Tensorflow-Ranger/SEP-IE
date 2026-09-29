def vowels_count(word):
    vowels = {'a', 'e', 'i', 'o', 'u'}
    count = 0
    for i, char in enumerate(word):
        if char in vowels:
            count += 1
        elif i == len(word) - 1 and char == 'y':
            count += 1
    return count
