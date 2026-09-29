def check_if_last_char_is_a_letter(txt):
    import re
    if not txt:
        return False
    last_char = txt[-1]
    if not last_char.isalpha():
        return False
    # Check if the last character is a single-letter word
    # Using regex to match a single alphabetic character at the end
    return bool(re.fullmatch(r'.*[^\w]*(\w)$', txt))
