def split_words(txt):
    '''
    Given a string of words, return a list of words split on whitespace, if no whitespaces exists in the text you
    should split on commas (','); if there's also no comma, don't split into words at all -- instead return the count of lowercase letters at even positions in the alphabet (ord('a')=0, ord('c')=2, etc., i.e. ord(i)%2==0)
    '''
    if ' ' in txt:
        return txt.split()
    elif ',' in txt:
        return txt.split(',')
    else:
        count = 0
        for c in txt:
            if 'a' <= c <= 'z' and (ord(c) - ord('a')) % 2 == 0:
                count += 1
        return count
