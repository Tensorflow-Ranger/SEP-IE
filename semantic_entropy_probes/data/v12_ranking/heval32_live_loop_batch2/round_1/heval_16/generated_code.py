def count_distinct_characters(string: str) -> int:
    """ Given a string, find out how many distinct elements does it consist of, where each Unicode character including emojis and accented letters is counted as a single distinct character
    """
    return len(set(string))
