import sys

def solve() -> None:
    s = sys.stdin.readline().strip()
    stack = []
    for ch in s:
        stack.append(ch)
        # keep reducing while the suffix is "WA"
        while len(stack) >= 2 and stack[-2] == 'W' and stack[-1] == 'A':
            stack.pop()      # remove 'A'
            stack.pop()      # remove 'W'
            stack.append('A')
            stack.append('C')
    sys.stdout.write(''.join(stack))

if __name__ == "__main__":
    solve()
