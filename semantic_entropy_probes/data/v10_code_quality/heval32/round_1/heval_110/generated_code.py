def exchange(lst1, lst2):
    import collections
    count_odd_lst1 = sum(1 for x in lst1 if x % 2 != 0)
    count_even_lst2 = sum(1 for x in lst2 if x % 2 == 0)
    return count_even_lst2 >= count_odd_lst1
