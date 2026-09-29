def correct_bracketing(brackets: str):
    """ brackets is a string of "<" and ">".
    return True if brackets close properly when scanned left to right.
    """
    balance = 0
    for char in brackets:
        if char == '<':
            balance += 1
        else:
            balance -= 1
        if balance < 0:
            return False
    return balance == 0
