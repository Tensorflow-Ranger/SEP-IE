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
    
    # We'll track the smoke positions at each time step
    # smoke_pos will be a set of (r,c) tuples where smoke is present at current time
    smoke_pos = {(0, 0)}
    
    result = []
    
    # Direction mappings
    dirs = {'N': (-1, 0), 'W': (0, -1), 'S': (1, 0), 'E': (0, 1)}
    
    for t in range(N):
        char = S[t]
        dr, dc = dirs[char]
        
        # Move all smoke according to wind
        new_smoke_pos = set()
        for (r, c) in smoke_pos:
            new_smoke_pos.add((r + dr, c + dc))
        
        # Check if (0,0) has smoke after movement
        if (0, 0) not in new_smoke_pos:
            new_smoke_pos.add((0, 0))
        
        smoke_pos = new_smoke_pos
        
        # Check if (R,C) has smoke at time t+0.5
        if (R, C) in smoke_pos:
            result.append('1')
        else:
            result.append('0')
    
    print(''.join(result))

if __name__ == "__main__":
    main()
