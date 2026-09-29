def tri(n):
    if n == 0:
        return [0]
    elif n == 1:
        return [0, 1]
    elif n == 2:
        return [0, 1, 1]
    result = [0, 1, 1]
    for i in range(3, n + 1):
        result.append(result[i-1] + result[i-2] + result[i-3])
    return result
