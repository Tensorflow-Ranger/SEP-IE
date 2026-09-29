def check_if_last_char_is_a_letter(txt):
    import re
    return bool(re.search(r'\b[a-zA-Z]\b$', txt))
