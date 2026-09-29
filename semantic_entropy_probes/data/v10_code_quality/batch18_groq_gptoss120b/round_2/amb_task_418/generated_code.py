from collections import deque

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    H = int(data[0])
    W = int(data[1])
    
    grid = []
    idx = 2
    for _ in range(H):
        grid.append(list(data[idx]))
        idx += 1
    
    A = int(data[idx]) - 1
    B = int(data[idx+1]) - 1
    C = int(data[idx+2]) - 1
    D = int(data[idx+3]) - 1
    idx += 4
    
    # Directions: up, down, left, right
    directions = [(-1,0), (1,0), (0,-1), (0,1)]
    
    # State in BFS: (row, col, kicks)
    queue = deque()
    visited = [[[False]*3 for _ in range(W)] for _ in range(H)]
    
    # Initially, we are at (A,B) with 0 kicks
    queue.append((A, B, 0))
    visited[A][B][0] = True
    
    while queue:
        r, c, kicks = queue.popleft()
        
        # If we reached the destination
        if r == C and c == D:
            print(kicks)
            return
        
        # Move actions: up to 2 steps in each of 4 directions
        for dr, dc in directions:
            for step in range(1, 3):
                nr = r + dr * step
                nc = c + dc * step
                if 0 <= nr < H and 0 <= nc < W:
                    if grid[nr][nc] == '#':
                        break  # Can't move through walls unless kicked
                    if not visited[nr][nc][kicks]:
                        visited[nr][nc][kicks] = True
                        queue.append((nr, nc, kicks))
                else:
                    break  # Out of bounds
        
        # Kick actions: up to 2 steps in each of 4 directions
        if kicks < 2:  # We can perform at most 2 kicks
            for dr, dc in directions:
                new_grid = [row[:] for row in grid]
                changed = False
                for step in range(1, 3):
                    nr = r + dr * step
                    nc = c + dc * step
                    if 0 <= nr < H and 0 <= nc < W:
                        if grid[nr][nc] == '#':
                            new_grid[nr][nc] = '.'
                            changed = True
                    # Continue even if out of bounds, but do nothing
                if changed:
                    # After kicking, we can move to adjacent cells that are now roads
                    # But we need to re-explore from current position with kicks+1
                    # So we just add the current position with kicks+1 to queue
                    if not visited[r][c][kicks+1]:
                        visited[r][c][kicks+1] = True
                        queue.append((r, c, kicks+1))
    
    # Should not reach here as per problem statement
    print(-1)

if __name__ == "__main__":
    main()
