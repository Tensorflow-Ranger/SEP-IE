from typing import List

def all_prefixes(string: str) -> List[str]:
    """ Return list of prefixes from shortest to longest of the input string, with prefixes cut at code-point boundaries, not grapheme boundaries, consistent with Python's standard string slicing. """
    return [string[:i] for i in range(len(string) + 1)]
