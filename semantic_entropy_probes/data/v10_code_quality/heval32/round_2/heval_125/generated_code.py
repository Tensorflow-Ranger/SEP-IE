def split_words(txt):
    import re
    words = re.split(r'\s+', txt.strip())
    if len(words) > 1:
        return words
    else:
        count = 0
        for char in txt:
            if char.islower():
                if (ord(char) - ord('a')) % 2 == 0:
                    count += 1
        return count
