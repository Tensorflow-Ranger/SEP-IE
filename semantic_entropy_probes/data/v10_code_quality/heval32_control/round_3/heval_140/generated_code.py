def fix_spaces(text):
    import re
    def replace(match):
        spaces = match.group(0)
        if len(spaces) > 2:
            return '-'
        else:
            return '_' * len(spaces)
    return re.sub(r' +', replace, text)
