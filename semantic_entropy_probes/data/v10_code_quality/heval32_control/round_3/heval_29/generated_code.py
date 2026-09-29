from typing import List

def filter_by_prefix(strings: List[str], prefix: str) -> List[str]:
    """ Filter an input list of strings for ones that contain a given prefix at the start of the string. Note that str.startswith('') is always True for any string, so an empty prefix matches every string in the input list; nothing in the reference restricts the prefix from being empty. Also, str.startswith is case-sensitive by default in the reference, and no case normalization is applied anywhere in the function. """
    return [s for s in strings if s.startswith(prefix)]
