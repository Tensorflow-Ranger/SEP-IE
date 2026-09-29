def is_bored(S):
    """
    You'll be given a string of words, and your task is to count the number
    of boredoms. Sentences are delimited by '.', '?' or '!'.
    """
    import re
    sentences = re.split('[.!?]', S)
    count = 0
    for sentence in sentences:
        if 'bored' in sentence:
            count += 1
    return count
