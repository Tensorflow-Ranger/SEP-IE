def count_distinct_characters(string: str) -> int:
    """ Given a string, find out how many distinct elements does it consist of, case-insensitive -- uppercase and lowercase versions of the same letter count as one distinct character (the string is lowercased before counting), yes -- the reference behavior lowercases the whole string then takes the set of characters, so any character (letters, digits, punctuation, whitespace) is counted as a distinct element the same way; there is no letter-only restriction"""
    return len(set(string.lower()))
