import sys

def solve() -> None:
    s = sys.stdin.readline().strip()
    stack = []
    for ch in s:
        stack.append(ch)
        # Resolve leftmost WA repeatedly (handled locally with stack)
        while len(stack) >= 2 and stack[-2] == 'W' and stack[-1] == 'A':
            # replace WA with AC
            stack.pop()      # remove 'A'
            stack.pop()      # remove 'W'
            stack.append('A')
            stack.append('C')
    sys.stdout.write(''.join(stack))

if __name__ == "__main__":
    solve()
