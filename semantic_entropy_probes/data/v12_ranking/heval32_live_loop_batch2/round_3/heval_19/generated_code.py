def sort_numbers(numbers: str) -> str:
    """ Input is a string of numberals from 'zero' to 'nine'.
    Valid choices are 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight' and 'nine'.
    There are no constraints on the order or frequency of repeated number words; they will all be sorted by their numeric value.
    If the input contains the same number word multiple times, all instances are preserved and sorted.
    Return the string with numbers sorted; the sort is stable—when two number words have the same numeric value, they maintain their original relative order from the input.
    """
    import re
    from collections import OrderedDict

    # Mapping of number words to their numeric values
    num_map = {
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

    # Sort the words based on their numeric value while maintaining stable sort
    sorted_words = sorted(words, key=lambda word: num_map[word])

    # Join the sorted words back into a string
    return ' '.join(sorted_words)
