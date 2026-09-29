def correct_bracketing(brackets: str):
    """ brackets is a string of "<" and ">".
    return True if brackets close properly. For very long inputs, the algorithm uses a simple linear scan with a depth counter, which is O(n) and has no recursion depth issues. Brackets cannot overlap; proper nesting requires that if '<' at position i matches '>' at position j, no other '<' at position k (i < k < j) can match '>' at position l (j < l). The exact definition of 'proper nesting' is: every '<' has a matching '>' in the correct order, equivalent to balanced parentheses.
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
