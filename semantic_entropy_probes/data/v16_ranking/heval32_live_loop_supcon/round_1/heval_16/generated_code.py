def count_distinct_characters(string: str) -> int:
    """ Given a string, find out how many distinct characters (case-insensitive) does it consist of
    """
    return len(set(string.lower()))
