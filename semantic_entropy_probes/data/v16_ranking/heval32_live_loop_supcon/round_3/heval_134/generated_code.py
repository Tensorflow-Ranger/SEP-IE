def check_if_last_char_is_a_letter(txt):
    import re
    if not txt:
        return False
    last_char = txt[-1]
    if not last_char.isalpha():
        return False
    words = re.split(r'\s+', txt.strip())
    if not words:
        return False
    last_word = words[-1]
    return len(last_word) == 1
