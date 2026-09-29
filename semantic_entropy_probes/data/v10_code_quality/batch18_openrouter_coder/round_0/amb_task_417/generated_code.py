```python
from collections import deque

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    H = int(data[0])
    W = int(data[1])
    S = data[2:2+H]
    A, B, C, D = map(int, data[2+H].split())
    
    # Convert to 0-indexed
    A -= 1
    B -= 1
    C -= 1
    D -= 1
    
    # Directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Precompute walls: grid[i][j] = 1 if wall, 0 if road
    grid = [[0]*W for _ in range(H)]
    for i in range(H):
        for j in range(W):
            if S[i][j] == '#':
                grid[i][j] = 1
                
    # BFS: state is (row, col, kicks)
    # We'll use 0-1 BFS since moving costs 0 and kicking costs 1
    from collections import deque
    queue = deque()
    # dist[kicks][row][col] - but we don't know max kicks
    # Instead, we track visited with (row, col, min_kicks_to_get_here)
    # Actually, better approach: for each (r,c,kicks), if we can reach it, 
    # we want the minimum kicks. So we can do BFS where we try all moves without kicks first,
    # then kicks which cost 1.
    
    # Let's do 0-1 BFS on states (row, col, kicks_used)
    # But potentially too many states. 
    # Better: BFS where state is (row, col) and we track min kicks to reach it.
    # When we move normally, kicks don't increase.
    # When we need to kick to make a path, kicks increase by 1.
    
    # For a given grid state (which depends on kicks used), what cells are reachable?
    # This is complex because kicking changes the grid.
    
    # Alternative idea:
    # Do BFS on (row, col, min_kicks) where we consider:
    # 1. All normal moves (cost 0 kicks) to reachable cells
    # 2. All possible kicks (cost 1 kick) to remove walls blocking paths
    
    # Let's think differently:
    # State: (row, col, kicks_used)
    # From each state, we can:
    # - Move to adjacent road cells (kicks unchanged)
    # - For each of 4 directions, kick (kicks + 1) and convert walls within 2 steps in that dir
    
    # To avoid recomputing grid state, note that if we've used K kicks, the set of walls
    # that could be cleared includes all walls within 2 steps in any of the 4 directions
    # from any position we've been in with <K kicks.
    
    # Even simpler: model this as a graph where nodes are (r,c) and edges have weight 0 or 1.
    # Weight 0: can move directly
    # Weight 1: need to kick to pass
    
    # Use 0-1 BFS (deque BFS): if cost 0, add to front; if cost 1, add to back
    
    # Let's define our graph properly:
    # Nodes: (r, c)
    # Edges:
    # 1. To adjacent (r',c') if it's '.', cost 0
    # 2. To (r',c') if it's '.' after kicking in some direction from current pos, cost 1
    
    # But when we kick, we change the grid. So adjacency depends on kick history.
    
    # Key insight: we don't need to track exact grid - just for each (r,c), what's the 
    # minimum number of kicks needed to make it passable.
    # Actually, even simpler: at any point, depending on kicks used so far, 
    # some walls are converted.
    
    # Easier modeling:
    # State space: (r, c, k) where k = min kicks to reach (r,c) in a state where all 
    # walls destroyed by <=k kicks are gone.
    # But transitions depend on which walls have been destroyed.
    
    # Simpler correct approach:
    # 0-1 BFS on graph with nodes (r,c) 
    # Edge (u,v) has weight 0 if v is adjacent and road in current kick-state
    # Edge has weight 1 if we need to kick to make path to v
    
    # But "current kick-state" is hard. What if we kicked earlier from a different place?
    
    # Best approach:
    # BFS with state (r, c, kicks) using double-ended queue (0-1 BFS)
    # From (r,c,kicks), we can:
    # 1. Move to adjacent road cells (same kicks) - add to front of queue
    # 2. Kick in any of 4 directions (kicks+1), then move through newly opened paths - add to back
    
    # However, tracking the exact state of the grid is complex.
    
    # Clever way:
    # When we're at (r,c) with k kicks used, we pretend all walls that could be 
    # destroyed by <=k kicks anywhere are destroyed.
    # But this overestimates available paths.
    
    # Correct approach:
    # Do 0-1 BFS where state is (r,c,kicks)
    # But we must simulate the actual destruction.
    
    # Let's implement a proper 0-1 BFS:
    # State: (r, c, kicks)
    # We keep track of min_kicks to reach each (r,c) - but that's not enough due to grid changes.
    
    # Final clean approach:
    # Model as shortest path with 0-1 edge weights.
    # Nodes are positions (r,c). 
    # For transition from (r,c):
    #   - to adjacent (nr,nc) if it's '.', weight 0  
    #   - else, we must kick. Try all 4 kick directions. 
    #     For each, we can destroy walls up to 2 steps away in that direction.
    #     Then we can move through those now-open cells, cost 1 total.
    
    # So from (r,c), for a kick in direction d:
    #   Destroy walls at (r+kd, c+ld) for k=1,2 in that direction
    #   Then we can move along that corridor until hitting an uncleared wall or boundary
    
    # But actually, we just want to get to (C,D). 
    # So we run 0-1 BFS from (A,B) to find min kicks to reach (C,D)
    
    # State: (r, c) with min_kicks to reach it
    # But again, transitions depend on kick history unless we precompute.
    
    # Clean solution:
    # Run 0-1 BFS. 
    # State: (r, c, kicks)
    # But we don't store full grid. Instead, when considering a kick, 
    # we see what walls it would clear from current position.
    
    # However, once a wall is cleared by a
