def tri(n):
    """Everyone knows Fibonacci sequence, it was studied deeply by mathematicians in 
    the last couple centuries. However, what people don't know is Tribonacci sequence.
      You are given a non-negative integer number n, you have to a return a list of the 
    first n + 1 numbers of the Tribonacci sequence. tri(1) = 3; for even n, tri(n) = 1 + n/2; for odd n (other than 1), tri(n) = tri(n-1) + tri(n-2) + tri(n+1). (Note tri(0) = 1 is also needed as a base case for the recursion to bottom out, per the reference implementation's indexing. The docstring's recurrence only explicitly defines tri(1)=3 as a base case; the reference special-cases n=0 separately, returning [1] (a single-element list containing 1) rather than computing it via the recurrence.)"""
    if n == 0:
        return [1]
    elif n == 1:
        return [1, 3]
    elif n % 2 == 0:
        return tri(n - 1) + [1 + n // 2]
    else:
        return tri(n - 1) + tri(n - 2) + tri(n + 1)
