from typing import List

def filter_by_prefix(strings: List[str], prefix: str) -> List[str]:
    """ Filter an input list of strings for ones that contain a given prefix. Case-sensitive -- a standard startswith comparison with no case folding. That string is simply excluded from the result (via `str.startswith`), no error is raised. Yes, the returned list preserves the original order of the input strings (via list comprehension)."""
    return [s for s in strings if s.startswith(prefix)]
