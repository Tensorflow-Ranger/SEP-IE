def exchange(lst1, lst2):
    from collections import Counter

    count_odd_lst1 = Counter(x % 2 for x in lst1 if x % 2 != 0)
    count_odd_lst2 = Counter(x % 2 for x in lst2 if x % 2 != 0)

    if count_odd_lst1[1] > count_odd_lst2[0]:
        return False

    return True
