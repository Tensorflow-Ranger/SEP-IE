def split_words(txt):
    '''
    Given a string of words, return a list of words split on whitespace, if no whitespaces exists in the text you
    should split on commas ',' if no commas exists you should return the number of lower-case letters with odd order in the
    alphabet (where odd order means ord(letter) - ord('a') is odd), ord('a') = 0, ord('b') = 1, ... ord('z') = 25
    '''
    if ' ' in txt:
        return txt.split()
    elif ',' in txt:
        return txt.split(',')
    else:
        count = 0
        for c in txt:
            if 'a' <= c <= 'z':
                if (ord(c) - ord('a')) % 2 == 1:
                    count += 1
        return count
