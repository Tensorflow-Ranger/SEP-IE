def largest_prime_factor(n: int):
    """Return the largest prime factor of n. Assume n > 1 and is not a prime.
    For a power of a prime, it returns that prime (e.g., for 8 it returns 2).
    """
    largest_factor = -1
    while n % 2 == 0:
        largest_factor = 2
        n //= 2
    i = 3
    while i * i <= n:
        while n % i == 0:
            largest_factor = i
            n //= i
        i += 2
    if n > 1:
        largest_factor = n
    return largest_factor
