def sort_numbers(numbers: str) -> str:
    """ Input is a string of numberals from 'zero' to 'nine'.
    Valid choices are 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight' and 'nine'.
    The input contains only valid, space-separated words from this list.
    Return the string with numbers sorted from smallest to largest.
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
    # Convert words to digits
    digits = [number_map[word] for word in numbers.split()]
    # Sort digits as integers
    digits_sorted = sorted(digits, key=lambda x: int(x))
    # Convert back to words and join
    return ' '.join([list(number_map.keys())[list(number_map.values()).index(d)] for d in digits_sorted])
