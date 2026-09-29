def tri(n):
    """Everyone knows Fibonacci sequence, it was studied deeply by mathematicians in 
    the last couple centuries. However, what people don't know is Tribonacci sequence.
      You are given a non-negative integer number n, you have to a return a list of the 
    first n + 1 numbers of the Tribonacci sequence. Not specifically addressed by the spec;
    the reference implementation computes the sequence iteratively from index 2 up to n, 
    building the list incrementally (this could be slow for very large n but isn't otherwise restricted)."""
    if n < 0:
        return []
    result = [0, 1]
    for i in range(2, n + 1):
        result.append(result[i-1] + result[i-2] + result[i-3])
    return result
