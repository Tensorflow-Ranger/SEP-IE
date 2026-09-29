def split_words(txt):
    '''
    Given a string of words, return a list of words split on whitespace, if no whitespaces exists in the text you
    should split on certain characters, if no such characters exists you should return the number of lower-case letters with odd order in the alphabet, ord('a') = 0, ord('b') = 1, ... ord('z') = 25
    '''
    import re
    if ' ' in txt:
        return txt.split()
    elif re.search(r'[,.!?:;]', txt):
        return re.split(r'[,.!?:;]', txt)
    else:
        count = 0
        for c in txt:
            if 'a' <= c <= 'z' and (ord(c) - ord('a')) % 2 == 1:
                count += 1
        return count
