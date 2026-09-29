def largest_prime_factor(n: int):
    """Return the largest prime factor of n. Assume n > 1 and is not a prime."""
    largest_factor = -1
    i = 2
    while i * i <= n:
        while n % i == 0:
            largest_factor = i
            n //= i
        i += 1
    if n > 1:
        largest_factor = n
    return largest_factor
