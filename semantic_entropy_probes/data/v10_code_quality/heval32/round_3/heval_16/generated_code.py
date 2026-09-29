def count_distinct_characters(string: str) -> int:
    """ Given a string, find out how many distinct elements does it consist of, where a distinct element is a unique character (case-insensitive) appearing in the string -- i.e. len(set(string.lower())). Yes; the string is lowercased before building the set, so 'A' and 'a' count as the same character. Treated as regular characters like any other -- included in the set and thus in the count. """
    return len(set(string.lower()))
