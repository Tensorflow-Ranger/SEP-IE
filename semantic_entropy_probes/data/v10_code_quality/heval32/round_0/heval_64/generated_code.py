def vowels_count(s):
    vowels = {'a', 'e', 'i', 'o', 'u', 'y'}
    return sum(1 for char in s.lower() if char in vowels)
