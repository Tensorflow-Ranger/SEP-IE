def check_if_last_char_is_a_letter(txt):
    if not txt:
        return False
    last_char = txt[-1].lower()
    return ord(last_char) >= 97 and ord(last_char) <= 122
