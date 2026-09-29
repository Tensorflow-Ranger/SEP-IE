def is_bore(S):
    import re
    sentences = re.split(r'[.?!]+', S)
    count = 0
    for sentence in sentences:
        if sentence.strip() and sentence[0:2] == 'I ':
            count += 1
    return count
