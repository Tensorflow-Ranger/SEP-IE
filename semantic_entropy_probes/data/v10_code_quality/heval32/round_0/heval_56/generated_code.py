def correct_bracketing(brackets: str):
    """ brackets is a string of "<" and ">".
    return True if brackets close properly.
    """
    balance = 0
    for bracket in brackets:
        if bracket == '<':
            balance += 1
        else:
            balance -= 1
        if balance < 0:
            return False
    return balance == 0
