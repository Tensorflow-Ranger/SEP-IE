def split_words(txt):
    '''
    Given a string of words, return a list of words split on whitespace, if no whitespaces exists in the text you
    should split on certain characters, if no such characters exists you should return the number of lower-case letters with odd order in the
    alphabet, ord('a') = 0, ord('b') = 1, ... ord('z') = 25. When no whitespace or splitting characters are found, the function should return an integer (the count), not a list containing an integer.
    '''
    import re
    if ' ' in txt:
        return txt.split()
    split_chars = re.compile(r'[,.:;!?]')
    if split_chars.search(txt):
        return split_chars.split(txt)
    count = 0
    for c in txt:
        if c.islower() and (ord(c) - ord('a')) % 2 == 1:
            count += 1
    return count
