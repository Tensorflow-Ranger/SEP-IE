import sys
from collections import deque
import heapq

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    H = int(data[idx]); idx += 1
    W = int(data[idx]); idx += 1
    
    F = []
    for i in range(H):
        row = []
        for j in range(W):
            row.append(int(data[idx])); idx += 1
        F.append(row)
    
    Q = int(data[idx]); idx += 1
    
    queries = []
    for _ in range(Q):
        A = int(data[idx]) - 1; idx += 1
        B = int(data[idx]) - 1; idx += 1
        Y = int(data[idx]); idx += 1
        C = int(data[idx]) - 1; idx += 1
        D = int(data[idx]) - 1; idx += 1
        Z = int(data[idx]); idx += 1
        queries.append((A, B, Y, C, D, Z))
    
    # Precompute for each floor level, which cells are accessible
    max_floor = max(max(row) for row in F)
    
    # For each floor x, we build a grid graph of cells where F[i][j] >= x
    # We'll do 0-1 BFS on the product graph: (row, col, floor)
    # But doing this for each query separately is too slow.
    # Instead, we use Dijkstra/0-1 BFS per query, but optimize by noting:
    # The state space is (i, j, floor), transitions:
    # 1. Stairs: (i,j,f) -> (i,j,f+1) or (i,j,f-1) cost 1
    # 2. Walkway: (i,j,f) -> (ni,nj,f) cost 0 if F[ni][nj] >= f
    
    def bfs_01(start_r, start_c, start_f, end_r, end_c, end_f):
        if (start_r, start_c, start_f) == (end_r, end_c, end_f):
            return 0
            
        # State: (cost, r, c, floor)
        dq = deque()
        # dist[r][c][f] = min cost to reach (r,c) at floor f
        # Since floors can be up to 1e6, we cannot store dist for all floors
        # Instead, we do BFS on the fly, keeping track of visited (r,c,f)
        # But even that is too much. 
        # Key insight: we only care about reaching the target floor at target cell.
        # So we run a 0-1 BFS in the state space (r, c, floor)
        
        # To avoid memory issues, we do BFS without storing all distances,
        # just track visited states and process until we reach destination
        
        from collections import deque
        visited = set()
        dq = deque()
        dq.append((0, start_r, start_c, start_f))
        visited.add((start_r, start_c, start_f))
        
        directions = [(-1,0),(1,0),(0,-1),(0,1)]
        
        while dq:
            cost, r, c, f = dq.popleft()
            
            if (r, c, f) == (end_r, end_c, end_f):
                return cost
                
            # Move via stairs (within same building)
            if f > 1:
                nf = f - 1
                if (r, c, nf) not in visited:
                    visited.add((r, c, nf))
                    dq.append((cost + 1, r, c, nf))
            if f < F[r][c]:
                nf = f + 1
                if (r, c, nf) not in visited:
                    visited.add((r, c, nf))
                    dq.append((cost + 1, r, c, nf))
                    
            # Move via walkways (same floor, adjacent cells)
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    if F[nr][nc] >= f:
                        if (nr, nc, f) not in visited:
                            visited.add((nr, nc, f))
                            dq.appendleft((cost, nr, nc, f))
                            
        return -1  # Should not happen per constraints
    
    results = []
    for A, B, Y, C, D, Z in queries:
        res = bfs_01(A, B, Y, C, D, Z)
        results.append(str(res))
        
    print('\n'.join(results))

if __name__ == "__main__":
    main()
