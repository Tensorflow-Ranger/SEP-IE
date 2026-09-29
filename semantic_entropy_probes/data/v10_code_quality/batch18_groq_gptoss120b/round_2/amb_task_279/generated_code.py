from collections import defaultdict
import sys

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
    
    # To store the positions of smoke particles with their creation time
    # We'll simulate backwards: for each time t, we'll see if (R,C) can reach (0,0) in t steps
    # But it's easier to track where the smoke from (0,0) ends up at time t
    
    # Let's track the position of the point that will end up at (R,C) at time t
    # If we reverse the process, starting from (R,C) at time t, and apply inverse wind,
    # we should end up at (0,0) at some time <= t for smoke to be present
    
    # For each time t, we want to know if smoke created at time i (1 <= i <= t) 
    # can reach (R,C) by time t+0.5
    
    # Better approach:
    # Track where the smoke that is at (0,0) initially ends up after each wind step
    # Then, at each time t, check if (R,C) is one of those positions
    
    # But we also generate new smoke at (0,0) every time there's no smoke there
    # So we need to track all current smoke positions
    
    # However, this might be too much. Let's think differently.
    
    # Key insight: simulate the movement of the "origin" point backwards
    # If we start from (R,C) and go backwards in time applying inverse moves,
    # and we land on (0,0) at time i where 1 <= i <= t, then smoke created at time i
    # will reach (R,C) at time t+0.5
    
    result = []
    
    # Precompute inverse moves
    inv_move = {'N': 'S', 'S': 'N', 'W': 'E', 'E': 'W'}
    inv_S = ''.join(inv_move[c] for c in S)
    
    # Now for each time t from 1 to N:
    # Start from (R,C) and apply inverse moves from t backwards to 1
    # If we reach (0,0) at some step i where 1 <= i <= t, then smoke was generated at time i
    # and reached (R,C) at time t+0.5
    
    # But we also need to ensure that when we go forward, the smoke doesn't get overwritten
    # Actually, let's do direct simulation but efficiently
    
    # Efficient way:
    # Track all current smoke positions in a set
    # At each time t:
    # 1. Move all smoke according to wind
    # 2. Check if (R,C) has smoke -> record result
    # 3. If (0,0) has no smoke, generate new smoke there
    
    # But this might be memory intensive. Let's optimize.
    
    # Since N can be up to 200000, we can't store all positions explicitly
    # But we notice that smoke only spreads from (0,0) and moves together
    # So at any time, the set of smoke positions is a translation of the initial point (0,0)
    # Wait, no. New smoke is generated at (0,0) whenever it's empty, so multiple smoke points can exist
    
    # Let's try a different approach:
    # For each time t, we want to know if (R,C) contains smoke at time t+0.5
    # This means:
    # 1. Smoke moved to (R,C) due to wind from previous positions
    # 2. Possibly new smoke was generated at (0,0)
    
    # Reverse simulation idea:
    # For each query time t, we trace back from (R,C) to see if it came from (0,0) 
    # in the first t steps through the wind movements
    
    # But we also have to account for new smoke generation
    
    # Let's do forward simulation with sets
    # Since coordinates are bounded by N, and N<=200000, storing positions is feasible
    
    current_smoke = set()
    result = []
    
    for t in range(N):
        wind = S[t]
        dr, dc = dir_map[wind]
        
        # Move all smoke
        new_smoke = set()
        for (r, c) in current_smoke:
            new_smoke.add((r + dr, c + dc))
            
        current_smoke = new_smoke
        
        # Check if (0,0) has smoke
        if (0,0) not in current_smoke:
            current_smoke.add((0,0))
            
        # Check if (R,C) has smoke now
        if (R,C) in current_smoke:
            result.append('1')
        else:
            result.append('0')
            
    print(''.join(result))

if __name__ == "__main__":
    main()
