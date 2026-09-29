def split_words(txt):
    '''
    Given a string of words, return a list of words split on whitespace, if no whitespaces exists in the text you
    should split on certain characters, if no such characters exists you should return the number of lower-case letters with odd order in the
    alphabet, ord('a') = 0, ord('b') = 1, ... ord('z') = 25, returned directly as a bare integer, not wrapped in a list.
    '''
    import re
    words = re.split(r'\s+', txt.strip())
    if len(words) > 1:
        return words
    split_chars = ['!', '.', ',', '?', ';', ':', '-', '_', '(', ')', '[', ']', '{', '}', '/', '\\', '|', '"', "'"]
    for char in split_chars:
        if char in txt:
            return txt.split(char)
    count = 0
    for c in txt:
        if 'a' <= c <= 'z' and (ord(c) - ord('a')) % 2 == 1:
            count += 1
    return count
