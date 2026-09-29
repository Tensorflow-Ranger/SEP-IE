def largest_prime_factor(n: int):
    """Return a large prime factor of n. Assume n > 1 and is not a prime. If n has multiple prime factors with the same maximum value (e.g., n = p²), the function returns that prime once, not its multiplicity. If n has a repeated prime factor (e.g., n = 4 = 2²), return the prime factor once: the function returns the largest distinct prime factor, not the highest power."""
    largest_factor = -1
    # Divide out factors of 2
    while n % 2 == 0:
        largest_factor = 2
        n = n // 2
    # Check for odd factors starting from 3
    i = 3
    while i * i <= n:
        while n % i == 0:
            largest_factor = i
            n = n // i
        i += 2
    # If n is a prime number greater than 2
    if n > 2:
        largest_factor = n
    return largest_factor
