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
    heap = [(0, A, B, 0)]  # (kicks, row, col, last_dir)
    
    while heap:
        kicks, r, c, last_dir = heapq.heappop(heap)
        
        if r == C and c == D:
            print(kicks)
            return
        
        if kicks > dist[r][c]:
            continue
        
        # Move to adjacent road cell
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] == '.' and dist[nr][nc] > kicks:
                dist[nr][nc] = kicks
                heapq.heappush(heap, (kicks, nr, nc, last_dir))
        
        # Front kick in all directions except last_dir
        for i in range(4):
            if i == last_dir:
                continue
            dr, dc = directions[i]
            for step in range(1, 3):
                nr, nc = r + dr * step, c + dc * step
                if 0 <= nr < H and 0 <= nc < W:
                    if grid[nr][nc] == '#':
                        grid[nr][nc] = '.'
                    if dist[nr][nc] > kicks + 1:
                        dist[nr][nc] = kicks + 1
                        heapq.heappush(heap, (kicks + 1, nr, nc, i))
    
    print(-1)

if __name__ == "__main__":
    main()
