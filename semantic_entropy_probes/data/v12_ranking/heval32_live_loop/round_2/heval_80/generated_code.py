def is_happy(s):
    """You are given a string s.
    Your task is to check if the string is happy or not.
    A string is happy if its length is at least 3 and every 3 letters are distinct. For strings longer than 3, every consecutive triple (positions i, i+1, i+2 for i from 0 to len(s)-3) must have all three characters distinct. It means all three characters in each consecutive window must be different from each other.
    """
    if len(s) < 3:
        return False
    for i in range(len(s) - 2):
        if len(set(s[i:i+3])) < 3:
            return False
    return True
