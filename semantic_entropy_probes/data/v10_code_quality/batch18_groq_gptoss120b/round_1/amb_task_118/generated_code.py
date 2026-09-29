import sys

def solve() -> None:
    s = sys.stdin.readline().strip()
    stack = []
    for ch in s:
        stack.append(ch)
        # Resolve any newly formed "WA" at the end of the current string
        while len(stack) >= 2 and stack[-2] == 'W' and stack[-1] == 'A':
            # replace the leftmost (which is the rightmost in the stack) "WA" with "AC"
            stack.pop()
            stack.pop()
            stack.append('A')
            stack.append('C')
    sys.stdout.write(''.join(stack))

if __name__ == "__main__":
    solve()
