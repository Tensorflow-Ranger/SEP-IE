def sort_numbers(numbers: str) -> str:
    """ Input is a string of numberals from 'zero' to 'nine'.
    Valid choices are 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight' and 'nine'.
    The input contains only valid, space-separated words from this list.
    Return the string with numbers sorted from smallest to largest.
    Duplicate words should maintain their relative input order (stable sort).
    """
    import re
    from collections import OrderedDict

    # Mapping of number words to their numeric values
    number_map = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
        'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9
    }

    # Split the input string into words
    words = numbers.split()

    # Sort the words based on their numeric value while maintaining stable sort
    sorted_words = sorted(words, key=lambda word: number_map[word])

    # Join the sorted words back into a string
    return ' '.join(sorted_words)
