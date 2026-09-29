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
    smoke.add((0, 0))

    result = []

    for i in range(N):
        direction = S[i]
        new_smoke = set()

        # Move existing smoke
        for (r, c) in smoke:
            if direction == 'N':
                new_r, new_c = r - 1, c
            elif direction == 'W':
                new_r, new_c = r, c - 1
            elif direction == 'S':
                new_r, new_c = r + 1, c
            elif direction == 'E':
                new_r, new_c = r, c + 1
            else:
                assert False, "Invalid direction"
            new_smoke.add((new_r, new_c))

        # Check if (R, C) is in new_smoke
        if (R, C) in new_smoke:
            result.append('1')
        else:
            result.append('0')

        # Regenerate smoke at (0, 0)
        new_smoke.add((0, 0))

        # Update smoke for next iteration
        smoke = new_smoke

    print(''.join(result))

if __name__ == "__main__":
    main()
