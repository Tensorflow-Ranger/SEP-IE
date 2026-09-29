import sys
from collections import defaultdict

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    R = int(data[1])
    C = int(data[2])
    S = data[3]
    
    # Map directions to coordinate changes
    dir_map = {'N': (-1, 0), 'W': (0, -1), 'S': (1, 0), 'E': (0, 1)}
    
    # Keep track of smoke positions at each time step
    # We'll track the smoke with a dictionary of positions
    smoke_positions = defaultdict(int)
    smoke_positions[(0, 0)] = 1
    
    result = []
    
    for t in range(N):
        char = S[t]
        dr, dc = dir_map[char]
        
        # Move all smoke
        new_smoke = defaultdict(int)
        for (r, c), count in smoke_positions.items():
            new_r = r + dr
            new_c = c + dc
            new_smoke[(new_r, new_c)] += count
        
        # Check if (0,0) has smoke after moving
        if new_smoke[(0, 0)] == 0:
            # Generate new smoke at (0,0)
            new_smoke[(0, 0)] += 1
            
        smoke_positions = new_smoke
        
        # Check if there's smoke at (R,C) at time t+0.5
        if smoke_positions[(R, C)] > 0:
            result.append('1')
        else:
            result.append('0')
    
    print(''.join(result))

if __name__ == "__main__":
    main()
