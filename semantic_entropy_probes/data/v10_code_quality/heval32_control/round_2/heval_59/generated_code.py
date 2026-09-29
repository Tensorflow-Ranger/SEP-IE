def largest_prime_factor(n: int):
    """Return a large prime factor of n. Assume n > 1 and is not a prime. Not applicable -- per the true spec, n is assumed to satisfy n > 1 and not be prime; negative n is outside the guaranteed input domain. 'Large prime factor' means the largest prime factor of n -- there is no tie to resolve, since a number's prime factorization is unique and has one well-defined maximum prime factor.
    """
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
