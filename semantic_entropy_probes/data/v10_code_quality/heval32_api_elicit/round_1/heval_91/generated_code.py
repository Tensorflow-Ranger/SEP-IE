def is_bored(S):
    """
    You'll be given a string of words, and your task is to count the number
    of boredoms. Sentences are delimited by '.', '?' or '!', and yes, case-sensitive;
    it must be capital "I" followed by a space, not lowercase "i".
    """
    import re
    sentences = re.split('[.!?]', S)
    count = 0
    for sentence in sentences:
        if 'I ' in sentence:
            count += 1
    return count
