def sort_numbers(numbers: str) -> str:
    """ Input is a string of numberals from 'zero' to 'nine'.
    Valid choices are 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight' and 'nine'.
    Return the string with numbers sorted; the sort is stable—when two number words have the same numeric value, they maintain their original relative order from the input.
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

    # Create a list of tuples with (numeric value, original word, original index)
    indexed_words = [(number_map[word], word, i) for i, word in enumerate(words)]

    # Sort by numeric value, then by original index to maintain stability
    sorted_words = sorted(indexed_words, key=lambda x: (x[0], x[2]))

    # Extract the sorted words
    sorted_words = [word for _, word, _ in sorted_words]

    return ' '.join(sorted_words)
