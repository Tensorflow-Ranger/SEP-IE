def tri(n):
    """Everyone knows Fibonacci sequence, it was studied deeply by mathematicians in 
    the last couple centuries. However, what people don't know is Tribonacci sequence.
      You are given a non-negative integer number n, you have to a return a list of the 
    first n + 1 numbers of the Tribonacci sequence.
    """
    if n == 0:
        return [0]
    elif n == 1:
        return [0, 1]
    trib = [0, 1]
    a, b = 0, 1
    for _ in range(2, n + 1):
        c = a + b + trib[-1]
        trib.append(c)
        a, b = b, c
    return trib
