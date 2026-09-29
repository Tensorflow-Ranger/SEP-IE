from typing import List

def filter_by_prefix(strings: List[str], prefix: str) -> List[str]:
    """ Filter an input list of strings for ones that contain a given prefix.
    The prefix is matched as a literal substring, which must be a contiguous sequence.
    """
    return [s for s in strings if s.startswith(prefix)]
