def tri(n):
    """Everyone knows Fibonacci sequence, it was studied deeply by mathematicians in 
    the last couple centuries. However, what people don't know is Tribonacci sequence.
      You are given a non-negative integer number n, you have to a return a list of the 
    first n + 1 numbers of the Tribonacci sequence. For odd n, use tri(n) = tri(n-1) + tri(n-2) + tri(n+1), where tri(n+1) refers to the value already computed in the forward-pass iteration (computed for the next even index); for even n, use tri(n) = 1 + n/2. The function should raise an exception if n is not an integer (e.g., if n is a float or string)."""
    if not isinstance(n, int) or n < 0:
        raise ValueError("n must be a non-negative integer")
    result = []
    for i in range(n + 1):
        if i == 0:
            result.append(0)
        elif i == 1:
            result.append(1)
        elif i % 2 == 0:
            result.append(1 + i // 2)
        else:
            result.append(result[i-1] + result[i-2] + result[i+1])
    return result
