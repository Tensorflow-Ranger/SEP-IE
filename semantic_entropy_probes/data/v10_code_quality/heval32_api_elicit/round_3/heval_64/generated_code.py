def vowels_count(s):
    vowels = "aeiouAEIOU"
    count = 0
    for i, char in enumerate(s):
        if char in vowels:
            if i == len(s) - 1 or char.lower() == 'y':
                count += 1
    return count
