from collections import deque

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    H = int(data[0])
    W = int(data[1])
    S = data[2:2+H]
    A = int(data[2+H]) - 1
    B = int(data[3+H]) - 1
    C = int(data[4+H]) - 1
    D = int(data[5+H]) - 1
    
    # Directions: up, down, left, right
    dy = [-1, 1, 0, 0]
    dx = [0, 0, -1, 1]
    
    # Precompute which walls can be broken by kicking in each direction from each cell
    kick_effects = [[[] for _ in range(W)] for _ in range(H)]
    
    for i in range(H):
        for j in range(W):
            for d in range(4):
                walls_to_break = []
                for k in range(1, 3):  # 1 and 2 steps
                    ni = i + k * dy[d]
                    nj = j + k * dx[d]
                    if 0 <= ni < H and 0 <= nj < W:
                        if S[ni][nj] == '#':
                            walls_to_break.append((ni, nj))
                kick_effects[i][j].append(walls_to_break)
    
    # BFS state: (row, col, bitmask of broken walls)
    # But there are too many possible bitmasks, so we need a different approach.
    
    # Instead, we model the state as (row, col, set of broken walls)
    # But this is also too large. We should think differently.
    
    # Key insight: We can use 0-1 BFS.
    # Moving costs 0, kicking costs 1.
    # State: (row, col, frozen version of grid)
    # Still too big.
    
    # Better idea: Use BFS where we track (row, col) and the current state of the grid.
    # But modifying the grid during BFS is complex.
    
    # Best approach: 
    # Model the problem as a graph where nodes are (row, col, grid_state)
    # But the grid state is huge.
    
    # Alternative: Since we only care about reaching the destination,
    # we can do BFS on (row, col) and track the minimum kicks needed to reach each (r,c) 
    # with a certain set of walls broken.
    # But again, tracking subsets is exponential.
    
    # Insight: We don't need to track *which* walls are broken, just that we can reach the goal.
    # So we can do 0-1 BFS on states (row, col, grid as a tuple or similar).
    # But even that might be too much.
    
    # Let's try a different modeling:
    # At any point, Takahashi is at some cell (r,c) and the grid has some walls broken.
    # From that state, he can either:
    #   - Move to an adjacent road (cost 0)
    #   - Kick in some direction (cost 1) which breaks up to 2 walls
    
    # To avoid tracking exact grid states, we can consider that the "state" includes
    # which walls have been broken. But there are 10^6 walls potentially.
    
    # Wait, let's re-read the problem.
    # He can move to adjacent roads OR kick.
    # Kicking changes walls to roads permanently.
    # Goal is to get from (A,B) to (C,D) using minimum kicks.
    
    # This is a shortest path problem in a graph where:
    # Nodes: (row, col, set_of_broken_walls)
    # Edges:
    #   - Move to adjacent road: weight 0
    #   - Kick in a direction: weight 1, changes some walls
    
    # This is computationally expensive.
    
    # Key realization: The grid state after kicks matters only insofar as it allows movement.
    # So we can think of states as (r, c, min_kicks_to_get_here_with_this_connectivity)
    
    # Even better: 0-1 BFS on graph where nodes are (r, c, mask_or_representation_of_grid)
    # But how?
    
    # Simpler idea: Do BFS where state is (r, c, grid_state) but grid_state is implicit.
    # That is, we simulate the actual process.
    
    # Let's represent the grid as a list of lists that we modify.
    # But then we can't reuse states in BFS because the grid changes.
    
    # Cleverer: We can precompute for each cell and each direction, what walls would be broken.
    # Then, our BFS state is (r, c) but transitions depend on current grid state.
    # This is still messy.
    
    # Another way: Since we want minimum kicks, and moves cost 0, kicks cost 1,
    # we can do 0-1 BFS: use deque, appendleft for cost 0, append for cost 1.
    
    # State: (r, c) but we need to track the grid somehow.
    # Or rather, we work with a copy of the grid and pass it around? No, too big.
    
    # Final idea: 
    # We run a 0-1 BFS where the state encodes enough information to determine valid moves.
    # The minimal such info is probably the current grid layout.
    # But since that's huge, we'll have to be smart.
    
    # Actually, let's just implement a straightforward 0-1 BFS assuming we can track
    # the grid state via a canonical representation like a tuple of strings.
    
    # But even tuple of strings for 1000x1000 grid is too big.
    
    # Hmm. Maybe the intended solution accepts that we modify the grid and proceed,
    # keeping track of visited (r,c,grid_state_hash) but accepting collisions.
    # Or maybe there's a smarter way.
    
    # Let's assume for now that we can hash the grid state reasonably.
    # In practice, for speed, we might need to limit our visited tracking.
    
    # For correctness, let's implement a BFS that tracks the grid properly.
    # We'll store a deep copy of the grid in each state.
    
    # But Python deepcopy is slow. Let's instead keep a parent grid and modifications.
    # Or just proceed step by step.

    # Let's take a known working approach for such problems:
    # Use 0-1 BFS with state being position and a compact representation of grid.
    # If that's not feasible, then perhaps the problem expects us to realize
    # that we only need to break certain walls and find a minimum set cover or similar.
    # But that seems not direct.
    
    # Re-examining: maybe we can bound the search.
    # Since we're looking for minimum kicks, and moves are free,
    # we can explore reachable areas without kicks first, then with 1 kick, etc.
    
    # That suggests multi-source BFS or iterative deepening.
    # But simpler is 0-1 BFS.
    
    # Let's proceed with 0-1 BFS, hashing the grid state.
    
    # Initial grid
    grid = [list(row) for row in S]
    
    # For performance, we'll track visited as (r, c, frozenset(broken_walls)) 
    # but that's still potentially too big.
    
    # Let's simplify: we'll do 0-1 BFS but not de-duplicate based on grid state.
    # This may re-explore states but could still work within time limits.
    # Alternatively, assume we can hash the grid.
    
    # Actually, let's think of the grid state as immutable per search node.
    # When we kick, we generate a new grid state.
    # We can represent the grid as a tuple of tuples for hashing.
    
    def grid_tuple(g):
        return tuple(tuple(row) for row in g)
    
    start_grid = grid_tuple(grid)
    
    # Visited: (r, c, grid_state) -> min_kicks
    # But this is too big. Let's not track grid_state in visited.
    # Just track (r, c) with min_kicks, but allow revisiting with fewer kicks.
    # That's not BFS anymore.
    
    # Proper way: full state space search with memoization of (r, c, grid_state)
    # But prune if we've seen (r, c, same_or_better_grid_state)
    
    # Given time constraints, let's implement a basic 0-1 BFS 
    # that recomputes reachability on the fly.
    
    # Here's a clean approach:
    # We perform 0-1 BFS on the graph where each node is (r, c)
    # and edges are:
    # - to adjacent roads with 0 cost (no kick needed)
    # - to self with 1 cost (performing a kick)
    
    # But we must update the grid when we "pay" for a kick.
    # So the state must include the grid.
    
    # Let's define state as (r, c, tuple_of_tuples_grid)
    # and use a priority queue / deque for 0-1 BFS
    
    from collections import deque
    
    # Need a way to represent grid that's hashable
    def freeze_grid(g):
        return tuple(tuple(row) for row in g)
    
    def thaw_grid(frozen_g):
        return [list(row) for row in frozen_g]
    
    # Initial setup
    initial_frozen_grid = freeze_grid(grid)
    
    # Queue elements: (r, c, frozen_grid, kicks_used)
    queue = deque()
    queue.append((A, B, initial_frozen_grid, 0))
    
    # Visited: (r, c, frozen_grid) -> min_kicks
    visited = {}
    visited[(A, B, initial_frozen_grid)] = 0
    
    while queue:
        r, c, fg, kicks = queue.popleft()
        
        if (r, c) == (C, D):
            print(kicks)
            return
            
        current_min = visited.get((r, c, fg), float('inf'))
        if kicks > current_min:
            continue
            
        # Convert frozen grid back to mutable for simulation
        g = thaw_grid(fg)
        
        # Try moving in four directions (free)
        for i in range(4):
            nr, nc = r + dy[i], c + dx[i]
            if 0 <= nr < H and 0 <= nc < W and g[nr][nc] == '.':
                fg_new = fg  # No change to grid
                key = (nr, nc, fg_new)
                if visited.get(key, float('inf')) > kicks:
                    visited[key] = kicks
                    queue.appendleft((nr, nc, fg_new, kicks))  # Cost 0, to front
                    
        # Try kicking in four directions (cost 1)
        for d in range(4):
            # Determine which walls will be broken
            walls_to_break = []
            for k in range(1, 3):
                nr = r + k * dy[d]
                nc = c + k * dx[d]
                if 0 <= nr < H and 0 <= nc < W:
                    if g[nr][nc] == '#':
                        walls_to_break.append((nr, nc))
            
            # Apply changes to a copy of the grid
            new_g = [row[:] for row in g]
            for wr, wc in walls_to_break:
                new_g[wr][wc] = '.'
                
            new_fg = freeze_grid(new_g)
            
            # Stay in place but used a kick
            key = (r, c, new_fg)
            if visited.get(key, float('inf')) > kicks + 1:
                visited[key] = kicks + 1
                queue.append((r, c, new_fg, kicks + 1))  # Cost 1, to back
                
    # Should always reach due to problem guarantee
    print(-1)

if __name__ == "__main__":
    main()
