from typing import List

def separate_paren_groups(paren_string: str) -> List[str]:
    paren_string = paren_string.replace(" ", "")
    groups = []
    current_group = []
    depth = 0
    for char in paren_string:
        if char == '(':
            depth += 1
            if depth == 1:
                current_group = []
        current_group.append(char)
        if depth == 0:
            groups.append(''.join(current_group))
        elif char == ')':
            depth -= 1
    return groups
