def largest_prime_factor(n: int):
    """Return the largest prime factor of n. Assume n > 1 and is not a prime.
    For a power of a prime, it returns that prime (e.g., for 8 it returns 2).
    The largest prime factor is the largest prime number that divides n, considering repeated primes only once.
    """
    largest_factor = -1
    # Divide n by 2 as long as it is divisible
    while n % 2 == 0:
        largest_factor = 2
        n = n // 2
    # n must be odd now, so we can skip even numbers
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
