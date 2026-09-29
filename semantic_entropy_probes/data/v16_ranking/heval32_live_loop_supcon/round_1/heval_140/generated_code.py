def fix_spaces(text):
    import re
    text = re.sub(' {2,}', '-', text)
    text = text.replace(' ', '_')
    return text
