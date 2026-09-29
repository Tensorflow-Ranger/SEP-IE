import sys

def solve() -> None:
    S = sys.stdin.readline().strip()
    stack = []
    for ch in S:
        stack.append(ch)
        # keep replacing leftmost WA which will be at the top after pushes
        while len(stack) >= 2 and stack[-2] == 'W' and stack[-1] == 'A':
            # replace WA with AC
            stack.pop()
            stack.pop()
            stack.append('A')
            stack.append('C')
    sys.stdout.write(''.join(stack))

if __name__ == "__main__":
    solve()
