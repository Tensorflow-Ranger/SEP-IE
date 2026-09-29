def is_bored(S):
    """
    You'll be given a string of words, and your task is to count the number
    of boredoms. Sentences are delimited by '.', '?' or '!', and a 'boredom'
    is specifically a sentence that starts with the word 'I' (as its own word,
    e.g. 'I am...').
    """
    import re
    sentences = re.split('[.!?]', S)
    count = 0
    for sentence in sentences:
        if sentence.strip() and sentence.strip().split()[0] == 'I':
            count += 1
    return count
