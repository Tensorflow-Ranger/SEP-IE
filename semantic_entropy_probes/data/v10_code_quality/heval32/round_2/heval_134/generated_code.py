def check_if_last_char_is_a_letter(txt):
    '''
    Create a function that returns True if the last character
    of a given string is an alphabetical character and is not
    closely associated with a word, precisely: the last "word" (the substring after the final space, per split(' ')) must have length exactly 1 and be alphabetic -- if the last space-separated token has more than one character, the last character is NOT standalone and the function returns False. Returns False in both cases: for an empty string, split(' ')[-1] is '' (length 0); for a whitespace-only or trailing-space string, the last segment after the final space is also empty. and False otherwise.
    '''
    parts = txt.split(' ')
    if not parts:
        return False
    last_part = parts[-1]
    if len(last_part) == 1 and last_part.isalpha():
        return True
    return False
