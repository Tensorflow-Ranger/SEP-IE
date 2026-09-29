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
    
    # Priority queue: (number of kicks, row, col, kick_count)
    pq = [(0, A, B, 0)]
    visited = [[[[float('inf')] * 4 for _ in range(W)] for _ in range(H)] for _ in range(4)]
    visited[0][A][B] = 0
    
    while pq:
        kicks, r, c, dir_mask = heapq.heappop(pq)
        
        if r == C and c == D:
            print(kicks)
            return
        
        # Move in any direction
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] == '.' and visited[0][nr][nc] > kicks:
                visited[0][nr][nc] = kicks
                heapq.heappush(pq, (kicks, nr, nc, dir_mask))
        
        # Front kick in each direction
        for d in range(4):
            if (dir_mask & (1 << d)) == 0:
                continue
            dr, dc = directions[d]
            for step in range(1, 3):
                nr, nc = r + dr * step, c + dc * step
                if 0 <= nr < H and 0 <= nc < W:
                    if grid[nr][nc] == '#':
                        grid[nr][nc] = '.'
                # After kick, we can move in that direction
                nr, nc = r + dr * step, c + dc * step
                if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] == '.' and visited[0][nr][nc] > kicks + 1:
                    visited[0][nr][nc] = kicks + 1
                    heapq.heappush(pq, (kicks + 1, nr, nc, dir_mask))
        
        # After kick, we can move in the direction of the kick
        for d in range(4):
            if (dir_mask & (1 << d)) == 0:
                continue
            dr, dc = directions[d]
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] == '.' and visited[0][nr][nc] > kicks + 1:
                visited[0][nr][nc] = kicks + 1
                heapq.heappush(pq, (kicks + 1, nr, nc, dir_mask))
    
    print(-1)

if __name__ == "__main__":
    main()
