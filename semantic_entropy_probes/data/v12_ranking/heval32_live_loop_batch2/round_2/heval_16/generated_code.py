def count_distinct_characters(string: str) -> int:
    """ Given a string, find out how many distinct elements does it consist of, where each Unicode character including emojis and accented letters is counted as a single distinct character; no; the function counts total characters in the string, including all repetitions of the same character
    """
    return len(set(string))
