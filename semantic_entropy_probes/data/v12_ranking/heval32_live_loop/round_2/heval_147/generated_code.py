def get_max_triples(n):
    from collections import defaultdict

    a = [i * i - i + 1 for i in range(1, n + 1)]
    mod_counts = defaultdict(int)
    for num in a:
        mod_counts[num % 3] += 1

    count = 0
    # Count combinations of (0, 0, 0)
    count += mod_counts[0] * (mod_counts[0] - 1) * (mod_counts[0] - 2) // 6
    # Count combinations of (1, 1, 1)
    count += mod_counts[1] * (mod_counts[1] - 1) * (mod_counts[1] - 2) // 6
    # Count combinations of (2, 2, 2)
    count += mod_counts[2] * (mod_counts[2] - 1) * (mod_counts[2] - 2) // 6
    # Count combinations of (0, 1, 2)
    count += mod_counts[0] * mod_counts[1] * mod_counts[2]
    return count
