def fix_spaces(text):
    """
    Given a string text, replace all spaces in it with underscores; only literal space characters ' ' count -- tabs, newlines, and other whitespace are not treated as spaces for this transformation. A run of exactly 2 consecutive spaces is replaced with two underscores (one underscore per space, via the 'end - start > 0' branch); only runs of MORE than 2 consecutive spaces get collapsed to a single '-'.
    """
    import re
    def replace_spaces(match):
        spaces = match.group(0)
        if len(spaces) > 2:
            return '-'
        else:
            return '_' * len(spaces)
    return re.sub(r' +', replace_spaces, text)
