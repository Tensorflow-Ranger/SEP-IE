def correct_bracketing(brackets: str):
    depth = 0
    for bracket in brackets:
        if bracket == '<':
            depth += 1
        else:
            depth -= 1
        if depth < 0:
            return False
    return depth == 0
