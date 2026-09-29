def correct_bracketing(brackets: str):
    """ brackets is a string of "<" and ">".
    return True if brackets close properly. A sequence where, scanning left to right, the running depth (incremented on '<', decremented on '>') never goes negative, and ends at exactly 0 -- proper nesting/ordering is required, not just equal counts.
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
