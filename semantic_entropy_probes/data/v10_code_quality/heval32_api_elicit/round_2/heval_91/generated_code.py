import re

def is_bored(S):
    """
    You'll be given a string of words, and your task is to count the number
    of boredoms. Sentences are delimited by '.', '?' or '!', and yes, case-sensitive;
    they act only as delimiters -- sentences are split on '.', '?', '!' via regex,
    and the delimiter characters themselves are consumed/stripped, not treated as part of sentence content;
    it must be capital "I" followed by a space, not lowercase "i".
    """
    sentences = re.split(r'[.!?]', S)
    count = 0
    for sentence in sentences:
        if sentence.strip().startswith('I '):
            count += 1
    return count
