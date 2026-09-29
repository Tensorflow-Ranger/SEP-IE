from typing import List

def separate_paren_groups(paren_string: str) -> List[str]:
    result = []
    current_group = []
    depth = 0
    for char in paren_string.replace(" ", ""):
        if char == '(':
            depth += 1
            if depth == 1:
                current_group = []
        current_group.append(char)
        if char == ')':
            depth -= 1
            if depth == 0:
                result.append(''.join(current_group))
    return result
