def largest_prime_factor(n: int):
    """Return a large prime factor of n. Assume n > 1 and is not a prime. It specifically means the LARGEST prime factor of n (the maximum j <= n such that j divides n and j is prime) -- not merely "a prime factor that is large" in some subjective sense."""
    largest_factor = -1
    while n % 2 == 0:
        largest_factor = 2
        n = n // 2
    i = 3
    while i * i <= n:
        while n % i == 0:
            largest_factor = i
            n = n // i
        i += 2
    if n > 1:
        largest_factor = n
    return largest_factor
