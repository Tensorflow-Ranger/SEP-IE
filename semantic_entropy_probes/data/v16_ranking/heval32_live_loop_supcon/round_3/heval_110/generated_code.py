def exchange(lst1, lst2):
    from collections import Counter

    count_odd_lst1 = Counter(x for x in lst1 if x % 2 != 0)
    count_even_lst2 = Counter(x for x in lst2 if x % 2 == 0)

    for odd, cnt in count_odd_lst1.items():
        if cnt > count_even_lst2[odd]:
            return "NO"
    return "YES"
