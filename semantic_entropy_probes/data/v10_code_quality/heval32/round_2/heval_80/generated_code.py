def is_happy(s):
    """You are given a string s.
    Your task is to check if the string is happy or not.
    A string is happy if its length is at least 3 and every 3 letters are distinct, yes, it checks every overlapping window of 3 consecutive letters (s[i], s[i+1], s[i+2] for each i), not just non-overlapping groups."""
    if len(s) < 3:
        return False
    for i in range(len(s) - 2):
        if len(set(s[i:i+3])) < 3:
            return False
    return True
