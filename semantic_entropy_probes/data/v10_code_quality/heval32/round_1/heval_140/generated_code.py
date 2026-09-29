def fix_spaces(text):
    """
    Given a string text, replace all spaces in it with underscores. Not applicable -- the function specifically checks for the literal space character " "; other whitespace characters (tabs, newlines) are treated as ordinary non-space characters and passed through unchanged.
    """
    return text.replace(' ', '_')
