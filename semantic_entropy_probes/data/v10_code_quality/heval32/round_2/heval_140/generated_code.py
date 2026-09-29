def fix_spaces(text):
    """
    Given a string text, replace all spaces in it with underscores. Not applicable; only the literal ' ' character (ASCII code point 32) is recognized as a space needing replacement. Other whitespace characters (tabs, newlines) are treated as ordinary non-space characters and passed through unchanged.
    """
    return text.replace(' ', '_')
