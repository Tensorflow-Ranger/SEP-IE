def sort_numbers(numbers: str) -> str:
    """ Input is a string of numberals from 'zero' to 'nine'.
    Valid choices are 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight' and 'nine'.
    Return the string with numbers sorted; no trailing delimiter; the output is just the sorted words joined by single spaces (see test outputs like 'zero four five seven eight nine'). Yes, duplicates are permitted and preserved in the output the same number of times they appear in the input.
    """
    number_words = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
        'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9
    }
    words = numbers.split()
    sorted_words = sorted(words, key=lambda x: number_words[x])
    return ' '.join(sorted_words)
