from typing import List

def all_prefixes(string: str) -> List[str]:
    """ Return list of prefixes from shortest to longest of the input string, with the shortest prefix being of length 1 (string[:1]) and the empty string (length 0) not included
    """
    return [string[:i] for i in range(1, len(string) + 1)]
