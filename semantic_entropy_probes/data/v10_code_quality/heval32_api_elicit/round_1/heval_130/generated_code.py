def tri(n):
    """Everyone knows Fibonacci sequence, it was studied deeply by mathematicians in 
    the last couple centuries. However, what people don't know is Tribonacci sequence.
      You are given a non-negative integer number n, you have to a return a list of the 
    first n + 1 numbers of the Tribonacci sequence. Yes, the returned list has n+1 elements total: tri(0) through tri(n) inclusive."""
    if n < 0:
        return []
    result = [0] * (n + 1)
    if n >= 0:
        result[0] = 0
    if n >= 1:
        result[1] = 1
    if n >= 2:
        result[2] = 1
    for i in range(3, n + 1):
        result[i] = result[i - 1] + result[i - 2] + result[i - 3]
    return result
