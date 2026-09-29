def largest_prime_factor(n: int):
    """Return a large prime factor of n. Assume n > 1 and is not a prime; not applicable -- input is guaranteed to satisfy 'n > 1 and is not a prime,' so no validation/error-raising is needed. For example, for n=27=3^3, the largest prime factor is simply 3 (confirmed by test case: 27 -> 3), regardless of the exponent/multiplicity."""
    largest_factor = -1
    # Divide n by 2 as long as it is divisible
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
