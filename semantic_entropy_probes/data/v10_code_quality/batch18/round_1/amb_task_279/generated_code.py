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

    # Use a dictionary to track the positions of smoke
    smoke = collections.defaultdict(int)
    smoke[(0, 0)] = 1  # Initially, there is smoke at (0, 0)

    result = []

    for t in range(N):
        direction = S[t]
        # Create a new dictionary for the next state
        next_smoke = collections.defaultdict(int)
        # Move smoke according to the wind direction
        for (r, c), count in smoke.items():
            if direction == 'N':
                nr, nc = r - 1, c
            elif direction == 'W':
                nr, nc = r, c - 1
            elif direction == 'S':
                nr, nc = r + 1, c
            elif direction == 'E':
                nr, nc = r, c + 1
            else:
                nr, nc = r, c
            next_smoke[(nr, nc)] += count
        # Generate new smoke at (0, 0)
        next_smoke[(0, 0)] += 1
        # Update smoke for next iteration
        smoke = next_smoke
        # Check if (R, C) has smoke at time t+0.5
        result.append('1' if smoke[(R, C)] > 0 else '0')

    print(''.join(result))

if __name__ == "__main__":
    main()
