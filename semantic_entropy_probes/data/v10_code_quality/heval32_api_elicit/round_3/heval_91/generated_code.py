import re

def is_bore(S):
    """
    You'll be given a string of words, and your task is to count the number
    of boredoms. Sentences are delimited by '.', '?' or '!', and yes, case-sensitive;
    they act only as delimiters -- sentences are split on each individual delimiter character (with any following whitespace consumed); consecutive delimiters produce empty sentence fragments between them, which simply fail the 'I ' check and don't affect the count;
    it must be capital "I" followed by a space, not lowercase "i".
    """
    sentences = re.split('([.!?])', S)
    count = 0
    for i in range(len(sentences)):
        if i % 2 == 1:  # check if it's a delimiter
            continue
        sentence = sentences[i].strip()
        if sentence == 'I ':
            count += 1
    return count
