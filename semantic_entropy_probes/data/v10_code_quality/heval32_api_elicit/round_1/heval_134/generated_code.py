def check_if_last_char_is_a_letter(txt):
    '''
    Create a function that returns True if the last character
    of a given string is an alphabetical character, only considering ASCII letters
    (checked via an ord() range of 97-122 after lowercasing), and is not
    closely associated with a word, and False otherwise.
    '''
    if not txt:
        return False
    last_char = txt[-1].lower()
    if not last_char.isalpha():
        return False
    return ord(last_char) >= 97 and ord(last_char) <= 122
