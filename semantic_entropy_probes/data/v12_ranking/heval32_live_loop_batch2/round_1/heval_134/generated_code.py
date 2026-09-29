def check_if_last_char_is_a_letter(txt):
    import re
    return bool(re.match(r'^[a-zA-Z]+$', txt) and txt[-1].isalpha())
