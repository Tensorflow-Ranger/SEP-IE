def sort_numbers(numbers: str) -> str:
    """ Input is a string of numberals from 'zero' to 'nine'.
    Valid choices are 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight' and 'nine'.
    Return the string with numbers sorted
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
    # Split the input string into individual number words
    words = numbers.split()
    # Convert each word to its corresponding digit
    digits = [number_map[word] for word in words]
    # Sort the digits
    digits.sort()
    # Join the sorted digits into a string and return
    return ''.join(digits)
