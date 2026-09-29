def check_if_last_char_is_a_letter(txt):
    import re
    words = re.split(r'\s+', txt.strip())
    if not words:
        return False
    last_word = words[-1]
    if len(last_word) == 0:
        return False
    if not last_word[-1].isalpha():
        return False
    return len(last_word) == 1
