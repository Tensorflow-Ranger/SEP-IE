import sys
import collections

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    R = int(data[1])
    C = int(data[2])
    S = data[3]

    # Use a deque to simulate the smoke spread efficiently
    smoke = collections.deque()
    smoke.append((0, 0))

    for t in range(N):
        # Move smoke according to the wind direction
        direction = S[t]
        new_smoke = set()
        for (r, c) in smoke:
            if direction == 'N':
                new_r, new_c = r - 1, c
            elif direction == 'W':
                new_r, new_c = r, c - 1
            elif direction == 'S':
                new_r, new_c = r + 1, c
            elif direction == 'E':
                new_r, new_c = r, c + 1
            new_smoke.add((new_r, new_c))
        smoke = new_smoke

        # Check if (R, C) has smoke after movement
        if (R, C) in smoke:
            print('Y', end='')
        else:
            print('N', end='')

        # Regenerate smoke at (0, 0) if needed
        if (R, C) == (0, 0):
            smoke.add((0, 0))

    print()

if __name__ == "__main__":
    main()
