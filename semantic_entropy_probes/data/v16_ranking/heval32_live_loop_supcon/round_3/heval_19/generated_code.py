def sort_numbers(numbers: str) -> str:
    """ Input is a string of numberals from 'zero' to 'nine'.
    Valid choices are 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight' and 'nine'.
    The input contains only valid, space-separated words from this list.
    Return the string with numbers sorted from smallest to largest in ascending order based on integer values (0-9).
    Duplicate words should maintain their relative input order (stable sort).
    """
    import re
    number_map = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9'
    }
    words = numbers.split()
    # Convert words to their numeric values for sorting
    sorted_words = sorted(words, key=lambda word: number_map[word])
    return ' '.join(sorted_words)
