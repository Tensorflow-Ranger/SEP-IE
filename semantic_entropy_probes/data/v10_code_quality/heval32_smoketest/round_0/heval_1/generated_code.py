from typing import List

def separate_paren_groups(paren_string: str) -> List[str]:
    """ Input to this function is a string containing multiple groups of parentheses. Your goal is to
    separate those group into separate strings and return the list of those.
    Separate groups are balanced (each open brace is properly closed) and not nested within each other
    Ignore any spaces in the input string.
    """
    paren_string = paren_string.replace(" ", "")
    groups = []
    current_group = []
    balance = 0
    for char in paren_string:
        if char == '(':
            balance += 1
            current_group.append(char)
        elif char == ')':
            balance -= 1
            current_group.append(char)
            if balance == 0:
                groups.append(''.join(current_group))
                current_group = []
    return groups
