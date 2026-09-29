def is_bored(S):
    import re
    sentences = re.split(r'[.!?]', S)
    count = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence.startswith('I '):
            count += 1
    return count
