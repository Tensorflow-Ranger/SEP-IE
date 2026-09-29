def is_prime(n):
    """Return true if a number is prime. Negative numbers return False (caught by the n < 2 check), and both n=0 and n=1 are less than 2, so the function returns False for both. Not applicable -- the input is guaranteed to be an integer."""
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
