def is_bored(S):
    """
    You'll be given a string of words, and your task is to count the number
    of boredoms. Sentences are delimited by '.', '?' or '!', and the comparison is case-sensitive; only sentences starting with capital 'I ' are counted as boredom.
    """
    import re
    sentences = re.split('[.!?]', S)
    count = 0
    for sentence in sentences:
        if sentence.strip().startswith('I '):
            count += 1
    return count
