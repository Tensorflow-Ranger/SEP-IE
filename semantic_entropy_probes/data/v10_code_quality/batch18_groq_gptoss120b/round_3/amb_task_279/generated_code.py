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
    
    # We'll track the position of the smoke that starts at (0,0)
    # Instead of tracking all smoke, we track where the original smoke would be at each step
    # and also account for newly generated smoke
    
    # Let's think reversely: instead of moving smoke forward, we can think of where the smoke 
    # at (R,C) at time t+0.5 came from.
    
    # But easier approach:
    # Track the position of the "smoke" that was at (0,0) at time 0
    # Each step, this position moves in the opposite direction of the wind
    # because we're tracing back where the smoke that ends up at (R,C) at time t came from
    
    # Actually, let's simulate the reverse process:
    # For each time t, we want to know if there's smoke at (R,C)
    # Smoke moves with the wind, so if we reverse the process,
    # we can see if (R,C) could have originated from (0,0)
    
    # Better approach:
    # Keep track of a set of positions where smoke could be
    # But N is up to 200000, so simulation might be too slow
    
    # Key insight:
    # Instead of tracking all smoke, track where the smoke that *could* end up at (R,C) came from
    # Run the process backwards: start at (R,C) and move opposite to wind directions
    # If at any time t, this backward path hits (0,0), then smoke reaches (R,C) at time t+0.5
    
    # So we simulate the point (R,C) moving in reverse wind directions
    # and check when it lands on (0,0)
    
    result = ['0'] * N
    
    # Start from (R,C)
    curr_r, curr_c = R, C
    
    # Go backwards in time from N to 1
    for t in range(N-1, -1, -1):
        # At time t+1, smoke moved according to S[t]
        # To reverse this, we move in the opposite direction
        move = S[t]
        if move == 'N':
            curr_r += 1
        elif move == 'W':
            curr_c += 1
        elif move == 'S':
            curr_r -= 1
        elif move == 'E':
            curr_c -= 1
            
        # After reversing the wind at time t+1, 
        # we check if this position was (0,0) at time t
        # That means smoke generated at time t reached (R,C) at time t+0.5
        if curr_r == 0 and curr_c == 0:
            result[t] = '1'
    
    print(''.join(result))

if __name__ == "__main__":
    main()
