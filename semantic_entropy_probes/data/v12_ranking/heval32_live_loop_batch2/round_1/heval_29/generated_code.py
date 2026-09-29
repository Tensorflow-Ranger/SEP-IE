from typing import List

def filter_by_prefix(strings: List[str], prefix: str) -> List[str]:
    """ Filter an input list of strings for ones that contain a given prefix. If the input list contains duplicate entries that match the prefix, all duplicates are retained in the output.
    """
    return [s for s in strings if s.startswith(prefix)]
