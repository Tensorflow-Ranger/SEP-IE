def tri(n):
    """Everyone knows Fibonacci sequence, it was studied deeply by mathematicians in 
    the last couple centuries. However, what people don't know is Tribonacci sequence.
      You are given a non-negative integer number n, you have to a return a list of the 
    first n + 1 numbers of the Tribonacci sequence. The function computes the Tribonacci sequence using: tri(1) = 3, tri(n) = 1 + n/2 for even n, and tri(n) = tri(n-1) + tri(n-2) + tri(n+1) for odd n. For even n, the formula uses exact regular division (n/2), producing floats like 2.0 for n=2, not integer division. The recurrence relation uses regular division (not integer division), producing floating-point results for even n values."""
    if n < 0:
        return []
    result = [0] * (n + 1)
    result[0] = 0
    if n >= 1:
        result[1] = 3
    for i in range(2, n + 1):
        if i % 2 == 0:
            result[i] = 1 + i / 2
        else:
            result[i] = result[i-1] + result[i-2] + result[i+1]
    return result
