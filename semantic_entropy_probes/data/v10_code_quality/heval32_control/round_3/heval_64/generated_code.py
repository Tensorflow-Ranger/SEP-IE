def vowels_count(s):
    vowels = {'a', 'e', 'i', 'o', 'u'}
    count = 0
    for char in s[:-1]:
        if char.lower() in vowels:
            count += 1
    if s[-1].lower() == 'y':
        count += 1
    return count
