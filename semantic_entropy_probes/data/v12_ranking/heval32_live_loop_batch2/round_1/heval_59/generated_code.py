def largest_prime_factor(n: int):
    """Return a large prime factor of n. Assume n > 1 and is not a prime. If n has multiple prime factors with the same maximum value (e.g., n = p²), the function returns that prime once, not its multiplicity."""
    import math
    largest_factor = -1
    while n % 2 == 0:
        largest_factor = 2
        n = n // 2
    i = 3
    max_factor = math.isqrt(n) + 1
    while i <= max_factor and n > 1:
        while n % i == 0:
            largest_factor = i
            n = n // i
            max_factor = math.isqrt(n) + 1
        i += 2
    if n > 1:
        largest_factor = n
    return largest_factor
