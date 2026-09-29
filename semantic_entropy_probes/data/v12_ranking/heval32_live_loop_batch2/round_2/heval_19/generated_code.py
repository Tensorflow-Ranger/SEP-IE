def sort_numbers(numbers: str) -> str:
    """ Input is a string of numberals from 'zero' to 'nine'.
    Valid choices are 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight' and 'nine'.
    If the input contains the same number word multiple times, all instances are preserved and sorted.
    Return the string with numbers sorted; the sort is stable—when two number words have the same numeric value, they maintain their original relative order from the input.
    """
    import re
    from collections import OrderedDict

    # Mapping of number words to their numeric values
    number_map = {
        'zero': 0,
        'one': 1,
        'two': 2,
        'three': 3,
        'four': 4,
        'five': 5,
        'six': 6,
        'seven': 7,
        'eight': 8,
        'nine': 9
    }

    # Split the input string into words
    words = numbers.split()

    # Create a list of tuples (number word, numeric value)
    numbered_words = [(word, number_map[word]) for word in words]

    # Sort the list by numeric value, preserving original order for equal values
    sorted_words = sorted(numbered_words, key=lambda x: x[1])

    # Extract the sorted number words
    sorted_words = [word for word, _ in sorted_words]

    # Join the sorted words into a single string
    return ' '.join(sorted_words)
