def count_distinct_characters(string: str) -> int:
    """ Given a string, find out how many distinct characters (case-insensitive) does it consist of
    Full-width and half-width forms of the same character are treated as distinct.
    """
    return len({c.lower() for c in string})
