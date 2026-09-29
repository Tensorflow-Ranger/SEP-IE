def is_happy(s):
    """You are given a string s.
    Your task is to check if the string is happy or not.
    A string is happy if its length is at least 3 and every 3 letters are distinct, meaning every run of 3 consecutive letters in the string must be distinct (i.e. for every window of 3 consecutive characters, no two of those three are the same) -- not that every possible group of any three letters in the string, consecutive or not, must be distinct."""
    if len(s) < 3:
        return False
    for i in range(len(s) - 2):
        if s[i] == s[i+1] or s[i] == s[i+2] or s[i+1] == s[i+2]:
            return False
    return True
