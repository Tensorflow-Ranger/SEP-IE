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
    
    INF = float('inf')
    dist = [[INF] * W for _ in range(H)]
    dist[A][B] = 0
    heap = [(0, A, B, 0)]  # (kicks, row, col, steps)
    
    while heap:
        kicks, r, c, steps = heapq.heappop(heap)
        
        if r == C and c == D:
            print(kicks)
            return
        
        if kicks > dist[r][c]:
            continue
        
        # Move to adjacent road cells
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] == '.' and dist[nr][nc] > kicks:
                dist[nr][nc] = kicks
                heapq.heappush(heap, (kicks, nr, nc, steps + 1))
        
        # Front kick in all four directions
        for dr, dc in directions:
            for step in range(1, 3):
                nr, nc = r + dr * step, c + dc * step
                if 0 <= nr < H and 0 <= nc < W:
                    if grid[nr][nc] == '#':
                        grid[nr][nc] = '.'  # Pretend it's a road
                # After 2 steps, we can move to the next cell
                if step == 2:
                    for nr, nc in [(r + dr, c + dc), (r - dr, c - dc)]:
                        if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] == '.' and dist[nr][nc] > kicks + 1:
                            dist[nr][nc] = kicks + 1
                            heapq.heappush(heap, (kicks + 1, nr, nc, steps + 1))
    
    print(-1)

if __name__ == "__main__":
    main()
