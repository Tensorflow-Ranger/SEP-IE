from typing import List

def all_prefixes(string: str) -> List[str]:
    """ Return list of prefixes from shortest to longest of the input string, with the shortest prefix being the first single character (length 1), not the empty string
    """
    return [string[:i] for i in range(1, len(string) + 1)]
