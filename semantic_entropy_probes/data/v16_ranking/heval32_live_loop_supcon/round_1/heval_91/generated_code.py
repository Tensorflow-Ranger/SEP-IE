def is_bored(S):
    """
    You'll be given a string of words, and your task is to count the number
    of boredoms. A boredom is a sentence that starts with the word "I".
    Sentences are delimited by '.', '?' or '!' (ignoring surrounding whitespace).
    """
    import re
    sentences = re.split(r'[.!?]', S.strip())
    count = 0
    for sentence in sentences:
        if sentence.strip().split()[0] == 'I':
            count += 1
    return count
