def get_max_triples(n):
    """
    You are given a positive integer n. You have to create an integer array a of length n.
        For each i (1 ≤ i ≤ n), the value of a[i] = i * i - i + 1.
        Return the number of triples (a[i], a[j], a[k]) of a where i < j < k, 
    and a[i] + a[j] + a[k] is a multiple of 3.
    """
    from collections import defaultdict

    a = [i*i - i + 1 for i in range(1, n+1)]
    mod_counts = defaultdict(int)
    for num in a:
        mod_counts[num % 3] += 1

    count = 0
    # All combinations of mod values that sum to 0 mod 3
    for mod1, count1 in mod_counts.items():
        for mod2, count2 in mod_counts.items():
            mod3 = (3 - mod1 - mod2) % 3
            count3 = mod_counts.get(mod3, 0)
            if mod1 <= mod2 <= mod3:
                if mod1 == mod2 == mod3:
                    count += count1 * count2 * count3 // 6
                elif mod1 == mod2 or mod2 == mod3 or mod1 == mod3:
                    count += count1 * count2 * count3 // 3
                else:
                    count += count1 * count2 * count3
    return count
