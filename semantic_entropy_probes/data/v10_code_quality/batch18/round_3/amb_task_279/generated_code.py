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

    # We track the positions of smoke using a set
    smoke = set()
    smoke.add((0, 0))  # Initial smoke at (0, 0)

    # We also track the movement of smoke using a deque for efficient shifts
    # Each step, we shift the smoke according to the direction
    # and then add new smoke at (0, 0)
    # We use a deque to represent the positions of smoke
    # and apply the direction to each position

    # To avoid recomputing, we can track the cumulative direction
    # and apply it to the initial position (0, 0)
    # Then, for each time step, we can compute the position of the smoke
    # that would have been generated at that time

    # We can track the cumulative direction vector
    dx = 0
    dy = 0
    for i in range(N):
        c = S[i]
        if c == 'N':
            dy -= 1
        elif c == 'S':
            dy += 1
        elif c == 'W':
            dx -= 1
        elif c == 'E':
            dx += 1

    # For each time step t, the smoke generated at t will be at (dx_t, dy_t)
    # where dx_t and dy_t are the cumulative direction up to time t
    # We can precompute the cumulative direction for each time step

    # Precompute cumulative direction
    cum_dx = [0] * (N + 1)
    cum_dy = [0] * (N + 1)
    for i in range(1, N + 1):
        c = S[i - 1]
        if c == 'N':
            cum_dy[i] = cum_dy[i - 1] - 1
        elif c == 'S':
            cum_dy[i] = cum_dy[i - 1] + 1
        elif c == 'W':
            cum_dx[i] = cum_dx[i - 1] - 1
        elif c == 'E':
            cum_dx[i] = cum_dx[i - 1] + 1

    # Now, for each time step t, the smoke generated at t will be at (cum_dx[t], cum_dy[t])
    # We need to check if (R, C) is equal to (cum_dx[t] + 0, cum_dy[t] + 0)
    # because the smoke generated at t is present at t+0.5

    res = []
    for t in range(1, N + 1):
        if (R == cum_dx[t] and C == cum_dy[t]):
            res.append('1')
        else:
            res.append('0')

    print(''.join(res))

if __name__ == "__main__":
    main()
