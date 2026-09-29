def fix_spaces(text):
    """
    Given a string text, replace all spaces in it with underscores. Guaranteed to always receive a string; no type-checking or error-handling for None/other types is required.
    """
    return text.replace(' ', '_')
