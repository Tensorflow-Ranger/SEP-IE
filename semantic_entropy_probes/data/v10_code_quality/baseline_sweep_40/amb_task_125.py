n = int(input())
strings = [input() for _ in range(n)]
strings.sort(key=lambda x: len(x))
print(''.join(strings))
