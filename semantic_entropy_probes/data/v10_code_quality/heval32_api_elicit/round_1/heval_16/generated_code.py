def count_distinct_characters(string: str) -> int:
    """ Given a string, find out how many distinct elements does it consist of, case-insensitive -- uppercase and lowercase versions of the same letter count as one distinct character (the string is lowercased before counting)"""
    return len(set(string.lower()))
