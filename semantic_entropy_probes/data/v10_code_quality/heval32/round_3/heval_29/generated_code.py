from typing import List

def filter_by_prefix(strings: List[str], prefix: str) -> List[str]:
    """ Filter an input list of strings for ones that contain a given prefix. Yes, case-sensitive; str.startswith performs an exact-case comparison with no normalization. Handled normally: an empty string only matches an empty prefix (str.startswith('') is always True); a non-empty prefix never matches an empty string. """
    return [s for s in strings if s.startswith(prefix)]
