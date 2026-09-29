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
    
    from collections import deque
    
    def bfs():
        dist = [[float('inf')] * W for _ in range(H)]
        dist[A][B] = 0
        q = deque([(A, B)])
        
        while q:
            x, y = q.popleft()
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] == '.' and dist[nx][ny] == float('inf'):
                    dist[nx][ny] = dist[x][y] + 1
                    q.append((nx, ny))
        
        return dist
    
    dist = bfs()
    
    if dist[C][D] == float('inf'):
        print(-1)
        return
    
    print(dist[C][D])

main()
