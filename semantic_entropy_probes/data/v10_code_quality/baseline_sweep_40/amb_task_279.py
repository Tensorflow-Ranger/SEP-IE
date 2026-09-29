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

    # Directions: N, W, S, E
    dir_map = {
        'N': (-1, 0),
        'W': (0, -1),
        'S': (1, 0),
        'E': (0, 1)
    }

    # Use a deque to simulate the movement of smoke
    smoke = collections.deque()
    smoke.append((0, 0))  # Start with smoke at (0, 0)

    result = []

    for t in range(N):
        direction = S[t]
        dr, dc = dir_map[direction]

        # Move all smoke
        new_smoke = []
        for (r, c) in smoke:
            nr, nc = r + dr, c + dc
            new_smoke.append((nr, nc))

        # Check if (R, C) has smoke at time t+0.5
        if (R, C) in new_smoke:
            result.append('1')
        else:
            result.append('0')

        # Remove smoke that moved out of bounds (not needed for this problem)
        # But we keep all for the next step
        smoke = new_smoke

    print(''.join(result))

if __name__ == "__main__":
    main()
