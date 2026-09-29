def is_bored(S):
    import re
    sentences = re.split(r'[.!?]', S)
    count = 0
    for sentence in sentences:
        if sentence.strip().startswith('I '):
            count += 1
    return count
