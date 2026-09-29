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
    
    # smoke_at[(r,c)] = set of times when smoke was generated that reaches (r,c)
    smoke_at = defaultdict(set)
    
    # Initially, no smoke
    # At each time t (1-based), smoke moves, then new smoke may be generated at (0,0)
    
    # We'll track where smoke that is generated at time t will be at time t+0.5
    # smoke_pos[t] = (r,c) where smoke generated at time t is located at time t+0.5
    
    # Let's reverse the process: for each time t, we compute where smoke that is at (R,C) at time t+0.5 came from at time t
    
    # Let's think differently:
    # At time t+0.5, smoke at (R,C) must have come from some position at time t
    # which is the reverse of the wind direction at time t
    
    # So, we can trace back from (R,C) at each time t+0.5 to see if it leads back to (0,0) at some time <= t
    
    # Let's track the position of (R,C) backwards in time
    # pos[t] = position at time t that will be at (R,C) at time t+0.5
    
    # Actually, let's think of it this way:
    # At time t+0.5, smoke at (R,C) must have been at (r_t, c_t) at time t
    # where (r_t, c_t) is (R,C) moved backwards by the wind at time t
    
    # So, (r_t, c_t) = (R,C) - wind_vector[t]
    # If (r_t, c_t) == (0,0), then smoke was generated there at time t
    
    result = []
    
    # For each time t, compute where (R,C) was at time t (before wind)
    r, c = R, C
    for t in range(N-1, -1, -1):
        # At time t+1, wind moves smoke
        # So to find where (R,C) was at time t+0.5 (after generation but before wind at t+1)
        # we move (R,C) backwards by wind at t+1
        # But we are iterating backwards, so for time t (0-based), wind is S[t]
        # We want to know where (r,c) was at time t+0.5 (after generation)
        # which is where it was before wind at time t+1 moved it to current (r,c)
        # So (r_prev, c_prev) + wind[t] = (r,c)
        # So (r_prev, c_prev) = (r,c) - wind[t]
        dr, dc = dir_map[S[t]]
        r_prev = r - dr
        c_prev = c - dc
        # Now (r_prev, c_prev) is where smoke was at time t+0.5 that ended up at (r,c) at time (t+1)+0.5
        # But we want to check if smoke was generated at (0,0) at time t+1
        # Smoke is generated at (0,0) at time t+1 if there was no smoke at (0,0) at time (t+1)
        # But we are checking where smoke that ends up at (R,C) came from
        # So if (r_prev, c_prev) == (0,0), then smoke generated at time t+1 reaches (R,C) at time (t+1)+0.5
        if r_prev == 0 and c_prev == 0:
            result.append('1')
        else:
            result.append('0')
        r, c = r_prev, c_prev
    
    # Since we built the result backwards, reverse it
    print(''.join(reversed(result)))

if __name__ == "__main__":
    main()
