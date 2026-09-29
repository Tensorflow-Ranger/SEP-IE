def is_bored(S):
    """
    You'll be given a string of words, and your task is to count the number
    of boredoms. A boredom is a sentence that starts with the word "I".
    Sentences are delimited by '.', '?' or '!' (ignoring surrounding whitespace).
    Sentences are split at every occurrence of '.', '?', or '!' regardless of context like abbreviations or numbers.
    """
    import re
    sentences = re.split(r'[.!?]', S)
    count = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and sentence.startswith('I'):
            count += 1
    return count
