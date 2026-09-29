def count_distinct_characters(string: str) -> int:
    """ Given a string, find out how many distinct elements does it consist of, where a 'distinct element' means a distinct character, counted case-insensitively (so 'A' and 'a' count as the same character)"""
    return len(set(char.lower() for char in string))
