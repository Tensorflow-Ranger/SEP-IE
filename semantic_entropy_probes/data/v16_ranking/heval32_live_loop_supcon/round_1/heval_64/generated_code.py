def vowels_count(word):
    vowels = {'a', 'e', 'i', 'o', 'u'}
    count = 0
    for i, char in enumerate(word):
        if char.lower() in vowels:
            count += 1
    if word and word[-1].lower() == 'y':
        count += 1
    return count
