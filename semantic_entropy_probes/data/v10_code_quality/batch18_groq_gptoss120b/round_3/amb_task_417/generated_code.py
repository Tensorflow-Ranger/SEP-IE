from collections import deque

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    H = int(data[0])
    W = int(data[1])
    S = data[2:2+H]
    A = int(data[2+H])
    B = int(data[3+H])
    C = int(data[4+H])
    D = int(data[5+H])
    
    # Convert to 0-indexed
    start = (A-1, B-1)
    goal = (C-1, D-1)
    
    # Preprocess the grid
    grid = []
    for i in range(H):
        grid.append(list(S[i]))
        
    # Directions: up, down, left, right
    directions = [(-1,0), (1,0), (0,-1), (0,1)]
    
    # BFS with state (row, col, kicks)
    # We want minimum kicks, so we can use 0-1 BFS
    # If we can move without kicking, cost 0
    # If we need to kick to proceed, cost 1
    
    # To do this properly, we'll use a deque and track visited states (r,c,kicks) 
    # But since we only care about minimum kicks, we can do BFS level by level by kicks.
    
    # Actually, let's think differently:
    # State: (row, col, min_kicks)
    # But that might be too much memory.
    
    # Let's do 0-1 BFS:
    # Use deque, append to left if cost 0, append to right if cost 1
    
    from collections import deque
    
    # Visited array: visited[r][c] = minimum kicks to reach (r,c)
    INF = 10**9
    visited = [[INF]*W for _ in range(H)]
    
    q = deque()
    # Start state
    q.append((0, start[0], start[1]))  # (kicks, r, c)
    visited[start[0]][start[1]] = 0
    
    while q:
        kicks, r, c = q.popleft()
        if kicks > visited[r][c]:
            continue
            
        # Try moving in 4 directions
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            # Check bounds and if it's a road
            if 0 <= nr < H and 0 <= nc < W:
                if grid[nr][nc] == '.':
                    # Can move for free
                    if kicks < visited[nr][nc]:
                        visited[nr][nc] = kicks
                        q.appendleft((kicks, nr, nc))
                else:
                    # It's a wall. Need to kick to make it passable.
                    # But we also need to check if we can kick to clear path ahead.
                    pass
            
            # Now simulate kicking in this direction
            # When we kick in direction (dr,dc), we can destroy walls up to 2 steps away
            # So cells: (r+dr,c+dc), (r+2*dr, c+2*dc) get turned from '#' to '.'
            
            # But we don't modify the original grid. Instead, when we try to move,
            # we consider that we could have kicked to make a path.
            
            # So for each direction, we can "pretend" that up to 2 steps ahead are cleared
            # This means from (r,c) trying to move in direction (dr,dc),
            # we can land at (r+dr,c+dc) or (r+2*dr, c+2*dc) if they were originally '#'
            # but we pay 1 kick cost
            
            # Let's check positions 1 and 2 steps ahead
            for step in [1,2]:
                nr_step = r + step * dr
                nc_step = c + step * dc
                if 0 <= nr_step < H and 0 <= nc_step < W:
                    # This cell would become '.' after kick
                    # So we can "move" here with 1 kick
                    new_kicks = kicks + 1
                    if new_kicks < visited[nr_step][nc_step]:
                        visited[nr_step][nc_step] = new_kicks
                        q.append((new_kicks, nr_step, nc_step))
                        
    print(visited[goal[0]][goal[1]])

if __name__ == "__main__":
    main()
