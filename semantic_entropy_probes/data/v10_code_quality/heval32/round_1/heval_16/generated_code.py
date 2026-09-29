def count_distinct_characters(string: str) -> int:
    """ Given a string, find out how many distinct elements does it consist of, where a distinct element is a unique character (case-insensitive) appearing in the string -- i.e. len(set(string.lower())). """
    return len(set(string.lower()))
