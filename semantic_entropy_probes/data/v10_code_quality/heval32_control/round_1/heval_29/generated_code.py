from typing import List

def filter_by_prefix(strings: List[str], prefix: str) -> List[str]:
    """ Filter an input list of strings for ones that contain a given prefix at the start of the string.
    """
    return [s for s in strings if s.startswith(prefix)]
