def split_words(txt):
    import re
    if ' ' in txt:
        return txt.split()
    elif re.search(r'[,.!?;:]', txt):
        return txt.split(', ')
    else:
        count = 0
        for c in txt:
            if 'a' <= c <= 'z' and (ord(c) - ord('a')) % 2 == 0:
                count += 1
        return count
