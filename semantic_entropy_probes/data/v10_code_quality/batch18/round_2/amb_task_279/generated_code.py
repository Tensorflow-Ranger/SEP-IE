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

    # Use a deque to simulate the spread of smoke
    smoke = collections.deque()
    smoke.append((0, 0))

    for t in range(N):
        direction = S[t]
        # Move all smoke in the current direction
        new_smoke = set()
        for (r, c) in smoke:
            if direction == 'N':
                new_r, new_c = r - 1, c
            elif direction == 'W':
                new_r, new_c = r, c - 1
            elif direction == 'S':
                new_r, new_c = r + 1, c
            else:  # E
                new_r, new_c = r, c + 1
            new_smoke.add((new_r, new_c))
        smoke = new_smoke

        # Check if (R, C) is in the smoke after the movement
        if (R, C) in smoke:
            print('1', end='')
        else:
            print('0', end='')

    print()

if __name__ == "__main__":
    main()
