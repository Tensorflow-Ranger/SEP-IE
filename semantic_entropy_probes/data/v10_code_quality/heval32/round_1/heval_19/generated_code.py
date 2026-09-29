def sort_numbers(numbers: str) -> str:
    """ Input is a string of numberals from 'zero' to 'nine'.
    Valid choices are 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight' and 'nine'.
    Return the string with numbers sorted numerically, by their numeric value (smallest to largest), not by alphabetical order of the words.
    """
    number_map = {
        'zero': '0',
        'one': '1',
        'two': '2',
        'three': '3',
        'four': '4',
        'five': '5',
        'six': '6',
        'seven': '7',
        'eight': '8',
        'nine': '9'
    }
    # Split the input string into words
    words = numbers.split()
    # Convert each word to its numeric value
    numeric_values = [number_map[word] for word in words]
    # Sort the words based on their numeric values
    sorted_words = [word for _, word in sorted(zip(numeric_values, words))]
    return ' '.join(sorted_words)
