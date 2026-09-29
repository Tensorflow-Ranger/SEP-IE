def sort_numbers(numbers: str) -> str:
    import re
    from collections import defaultdict

    number_words = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9'
    }

    words = re.findall(r'\b\w+\b', numbers)
    mapped = [number_words[word] for word in words if word in number_words]
    mapped.sort(key=lambda x: int(x))
    return ' '.join(mapped)
