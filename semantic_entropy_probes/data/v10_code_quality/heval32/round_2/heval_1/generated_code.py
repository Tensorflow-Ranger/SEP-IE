from typing import List

def separate_paren_groups(paren_string: str) -> List[str]:
    paren_string = paren_string.replace(" ", "")
    result = []
    depth = 0
    start = 0
    for i, char in enumerate(paren_string):
        if char == '(':
            depth += 1
            if depth == 1:
                start = i
        elif char == ')':
            depth -= 1
            if depth == 0:
                result.append(paren_string[start:i+1])
    return result
