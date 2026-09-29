def fix_spaces(text):
    """
    Given a string text, replace all spaces in it with underscores; consecutive spaces are each replaced individually with underscores; so two spaces become two underscores '__'; the solution uses a linear scan with explicit space-counting logic; it handles long strings efficiently; no, only ASCII spaces (0x20); non-breaking spaces and other Unicode spaces are not replaced.
    """
    result = []
    space_count = 0
    for char in text:
        if char == ' ':
            space_count += 1
        else:
            if space_count > 0:
                result.append('_' * space_count)
                space_count = 0
            result.append(char)
    if space_count > 0:
        result.append('_' * space_count)
    return ''.join(result)
