import sys
import heapq

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    H = int(data[0])
    W = int(data[1])
    grid = data[2:2+H]
    A = int(data[2+H])
    B = int(data[3+H])
    C = int(data[4+H])
    D = int(data[5+H])
    
    A -= 1
    B -= 1
    C -= 1
    D -= 1
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Convert grid to 0-based
    grid = [list(row) for row in grid]
    
    # Priority queue: (num_kicks, row, col)
    pq = [(0, A, B)]
    # Visited: (row, col) -> min_kicks
    visited = {}
    
    while pq:
        kicks, r, c = heapq.heappop(pq)
        
        if (r, c) == (C, D):
            print(kicks)
            return
        
        if (r, c) in visited and visited[(r, c)] <= kicks:
            continue
        
        visited[(r, c)] = kicks
        
        # Move to adjacent road cells
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] == '.' and (nr, nc) not in visited or visited[(nr, nc)] > kicks:
                heapq.heappush(pq, (kicks, nr, nc))
        
        # Front kick in each direction
        for dr, dc in directions:
            # Kick up to 2 steps in direction
            for step in range(1, 3):
                nr, nc = r + dr * step, c + dc * step
                if 0 <= nr < H and 0 <= nc < W:
                    if grid[nr][nc] == '#':
                        grid[nr][nc] = '.'  # Temporarily mark as road
                    # Push new state with one more kick
                    heapq.heappush(pq, (kicks + 1, nr, nc))
    
    print(-1)

if __name__ == "__main__":
    main()
