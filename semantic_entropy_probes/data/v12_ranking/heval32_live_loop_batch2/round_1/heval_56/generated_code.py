def correct_bracketing(brackets: str):
    """ brackets is a string of "<" and ">".
    return True if brackets close properly. For very long inputs, the algorithm uses a simple linear scan with a depth counter, which is O(n) and has no recursion depth issues.
    """
    depth = 0
    for char in brackets:
        if char == '<':
            depth += 1
        elif char == '>':
            depth -= 1
            if depth < 0:
                return False
    return depth == 0
