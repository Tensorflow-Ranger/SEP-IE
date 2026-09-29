def exchange(lst1, lst2):
    """In this problem, you will implement a function that takes two lists of numbers,
    and determines whether it is possible to perform exchanges to make lst1 
    a list of only even numbers. If either lst1 or lst2 is empty, the function should proceed normally: if lst1 is empty it already has no odd numbers (return "YES"), if lst2 is empty there are no even numbers to exchange with. Elements can be exchanged between lst1 and lst2 to achieve this goal.
    If it is possible to make all the elements of lst1 even through exchanges, 
    return "YES". Otherwise, return "NO".
    """
    if not lst1:
        return "YES"
    if not lst2:
        return "NO"
    odd_count = sum(1 for x in lst1 if x % 2 != 0)
    even_count = sum(1 for x in lst2 if x % 2 == 0)
    return "YES" if odd_count <= even_count else "NO"
