def split_words(txt):
    '''
    Given a string of words, return a list of words split on whitespace, if no whitespaces exists in the text you
    should split on certain characters, if no such characters exists you should return the number of lower-case letters with odd order in the alphabet, ord('a') = 0, ord('b') = 1, ... ord('z') = 25. All occurrences are counted; each appearance of a letter with odd-order (even ord value) is counted separately. Only ASCII lowercase letters a-z are considered in the letter-counting fallback; Unicode letters are not addressed. When no whitespace or splitting characters are found, the function should return an integer (the count), not a list containing an integer.
    '''
    import re
    words = re.split(r'\s+', txt.strip())
    if len(words) > 1:
        return words
    split_chars = ['!', '.', ',', '?', ';', ':', '-', '_', '(', ')', '[', ']', '{', '}', '/', '\\', '|', '`', '~']
    for char in split_chars:
        if char in txt:
            return txt.split(char)
    count = 0
    for c in txt:
        if 'a' <= c <= 'z':
            if (ord(c) - ord('a')) % 2 == 0:
                count += 1
    return count
