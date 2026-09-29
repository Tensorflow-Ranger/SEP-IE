def tri(n):
    """Everyone knows Fibonacci sequence, it was studied deeply by mathematicians in 
    the last couple centuries. However, what people don't know is Tribonacci sequence.
      You are given a non-negative integer number n, you have to a return a list of the 
    first n + 1 numbers of the Tribonacci sequence. tri(1) = 3; for even n, tri(n) = 1 + n/2; for odd n (other than 1), tri(n) = tri(n-1) + tri(n-2) + tri(n+1). (Note tri(0) = 1 is also needed as a base case for the recursion to bottom out, per the reference implementation's indexing.)"""
    if n == 0:
        return [1]
    elif n == 1:
        return [1, 3]
    res = [1, 3]
    for i in range(2, n + 1):
        if i % 2 == 0:
            res.append(1 + i // 2)
        else:
            res.append(res[-1] + res[-2] + res[-3])
    return res
