def is_happy(s):
    """You are given a string s.
    Your task is to check if the string is happy or not.
    A string is happy if its length is at least 3 and every 3 letters are distinct, meaning every run of 3 consecutive letters in the string must be distinct (i.e. for every window of 3 consecutive characters, no two of those three are the same). Same rule as for longer strings: the reference checks each consecutive window of 3 characters and requires all three to be pairwise different. For a 3-character string there's exactly one such window (the whole string), which must have 3 distinct characters."""
    if len(s) < 3:
        return False
    for i in range(len(s) - 2):
        if s[i] == s[i+1] or s[i] == s[i+2] or s[i+1] == s[i+2]:
            return False
    return True
