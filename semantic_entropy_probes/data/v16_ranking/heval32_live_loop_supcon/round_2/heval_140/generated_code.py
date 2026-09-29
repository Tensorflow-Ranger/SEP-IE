def fix_spaces(text):
    import re
    result = []
    i = 0
    while i < len(text):
        if text[i] == ' ':
            count = 0
            while i < len(text) and text[i] == ' ':
                count += 1
                i += 1
            if count > 2:
                result.append('-')
            else:
                result.append('_' * count)
        else:
            result.append(text[i])
            i += 1
    return ''.join(result)
