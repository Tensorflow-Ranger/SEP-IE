def sort_numbers(numbers: str) -> str:
    """ Input is a string of numberals from 'zero' to 'nine', not applicable -- input is guaranteed to be a well-formed space-delimited string of the valid number words. Valid choices are 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight' and 'nine'.
    Return the string with numbers sorted numerically, by their numeric value (smallest to largest), not by alphabetical order of the words. Yes; the output is the sorted words joined by single spaces.
    """
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
    words = numbers.split()
    sorted_words = sorted(words, key=lambda x: number_map[x])
    return ' '.join(sorted_words)
