def correct_bracketing(brackets: str):
    depth = 0
    for char in brackets:
        if char == '<':
            depth += 1
        else:
            depth -= 1
        if depth < 0:
            return False
    return depth == 0
