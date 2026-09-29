def correct_bracketing(brackets: str):
    """ brackets is a string of "<" and ">".
    return True if brackets close properly, yes, order matters: the running depth must never go negative (a '>' cannot appear before its matching '<'), not just have equal total counts.
    """
    depth = 0
    for char in brackets:
        if char == '<':
            depth += 1
        else:
            depth -= 1
        if depth < 0:
            return False
    return depth == 0
