def fix_spaces(text):
    import re
    result = []
    i = 0
    while i < len(text):
        if text[i] == ' ':
            j = i
            while j < len(text) and text[j] == ' ':
                j += 1
            spaces = j - i
            if spaces > 2:
                result.append('-')
            else:
                result.append('_' * spaces)
            i = j
        else:
            result.append(text[i])
            i += 1
    return ''.join(result)
