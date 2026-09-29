def fix_spaces(text):
    import re
    def replace_spaces(match):
        spaces = match.group(0)
        if len(spaces) > 2:
            return '-'
        else:
            return '_' * len(spaces)
    return re.sub(r' ', replace_spaces, text)
