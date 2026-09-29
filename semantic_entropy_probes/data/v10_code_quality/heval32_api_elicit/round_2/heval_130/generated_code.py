def tri(n):
    """Everyone knows Fibonacci sequence, it was studied deeply by mathematicians in 
    the last couple centuries. However, what people don't know is Tribonacci sequence.
      You are given a non-negative integer number n, you have to a return a list of the 
    first n + 1 numbers of the Tribonacci sequence. Yes, the returned list has n+1 elements total: tri(0) through tri(n) inclusive. Not specified; no input validation or exception-raising is required -- the input is guaranteed valid per the docstring."""
    if n == 0:
        return [0]
    elif n == 1:
        return [0, 0]
    elif n == 2:
        return [0, 0, 0]
    trib = [0, 0, 0]
    for i in range(3, n + 1):
        trib.append(trib[i-1] + trib[i-2] + trib[i-3])
    return trib
