def check_if_last_char_is_a_letter(txt):
    import re
    parts = txt.split(' ')
    if len(parts) == 0:
        return False
    last_part = parts[-1]
    if len(last_part) == 1 and re.match(r'[a-zA-Z]', last_part):
        return True
    return False
