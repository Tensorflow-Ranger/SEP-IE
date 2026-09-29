def is_happy(s):
    """You are given a string s.
    Your task is to check if the string is happy or not.
    A string is happy if its length is at least 3 and every 3 letters are distinct, with non-alphabetic characters (digits, symbols, whitespace) included in the distinctness check as regular characters. Yes, each 3-character window must contain three distinct characters (s[i] != s[i+1], s[i+1] != s[i+2], and s[i] != s[i+2])."""
    if len(s) < 3:
        return False
    for i in range(len(s) - 2):
        if s[i] == s[i+1] or s[i+1] == s[i+2] or s[i] == s[i+2]:
            return False
    return True
