from typing import List

def filter_by_prefix(strings: List[str], prefix: str) -> List[str]:
    """ Filter an input list of strings for ones that contain a given prefix. Yes -- "start with" means the string must begin with prefix (str.startswith), not merely contain it anywhere.
    """
    return [s for s in strings if s.startswith(prefix)]
