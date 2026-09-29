def sort_numbers(numbers: str) -> str:
    """ Input is a string of numberals from 'zero' to 'nine'.
    Valid choices are 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight' and 'nine'.
    Return the string with numbers sorted in ascending order, from smallest to largest. Not applicable in a special sense -- inputs are number-words ('zero' through 'nine'); ties (equal values) simply keep their original relative order under a standard stable sort, with no additional tie-breaking rule needed.
    """
    import re
    number_words = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9'
    }
    # Split the input string into words
    words = numbers.split()
    # Replace each word with its corresponding digit
    digit_list = [number_words[word] for word in words]
    # Convert to integers for sorting
    int_list = list(map(int, digit_list))
    # Sort the list of integers while preserving the original order for ties
    # Using a stable sort by converting to tuples of (value, index)
    indexed_list = [(int_list[i], i) for i in range(len(int_list))]
    indexed_list.sort()
    # Extract the sorted values and convert back to words
    sorted_values = [str(val) for val, _ in indexed_list]
    # Map back to words
    sorted_words = [list(number_words.keys())[list(number_words.values()).index(val)] for val in sorted_values]
    return ' '.join(sorted_words)
