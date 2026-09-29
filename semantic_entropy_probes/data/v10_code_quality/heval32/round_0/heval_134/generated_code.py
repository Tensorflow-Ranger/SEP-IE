def check_if_last_char_is_a_letter(txt):
    '''
    Create a function that returns True if the last character
    of a given string is an alphabetical character and is not
    closely associated with a word, and False otherwise.
    '''
    if not txt:
        return False
    last_char = txt[-1]
    if not last_char.isalpha():
        return False
    return not (last_char in 'smt')
