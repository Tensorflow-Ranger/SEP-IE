def is_bored(S):
    """
    You'll be given a string of words, and your task is to count the number
    of boredoms. Sentences are delimited by '.', '?' or '!', and a boredom is a sentence that starts with the word 'I ' (uppercase I followed by a space).
    """
    import re
    sentences = re.split('[.!?]', S)
    count = 0
    for sentence in sentences:
        if sentence.strip().startswith('I '):
            count += 1
    return count
