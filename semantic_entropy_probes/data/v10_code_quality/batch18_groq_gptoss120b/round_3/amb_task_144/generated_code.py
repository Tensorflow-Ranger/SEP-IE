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
    
    # Precompute for each floor level the connected components (blocks that have >= that floor)
    # We'll use 0-indexing for blocks
    max_floor = 0
    for i in range(H):
        for j in range(W):
            max_floor = max(max_floor, F[i][j])
    
    # For efficiency, we will run 0-1 BFS for each query.
    # State: (cost, row, col, floor)
    # Transitions:
    # 1. Move up/down within building: cost +1 (stair move)
    # 2. Move to adjacent building with same floor via walkway: cost +0
    
    def bfs_01(s_r, s_c, s_f, t_r, t_c, t_f):
        if (s_r, s_c, s_f) == (t_r, t_c, t_f):
            return 0
            
        # To avoid recomputing for every pair, we can do a reverse approach:
        # But since queries are separate, we must compute per query.
        # Use 0-1 BFS: deque where we appendleft for cost 0, append for cost 1
        
        # State: (cost, r, c, f)
        dq = deque()
        # dist[r][c][f] = min cost to reach floor f in block (r,c)
        # Since f can be up to 1e6, we can't store all distances.
        # Instead, we run 0-1 BFS for each query separately.
        
        # Let's define reachable floors per block dynamically.
        # But again, due to memory constraints, let's proceed carefully.
        
        # Actually, since F[i][j] <= 1e6 and H,W<=500, storing dist for all (r,c,f) is too much.
        # So we must run BFS per query.
        
        # For this query, we consider states (r, c, f) where 1 <= f <= F[r][c]
        # We need to limit search space somehow.
        # Key insight: we only care about floors that are present in at least one block.
        # But even then, it's still too broad.
        
        # Alternative idea: model as graph where nodes are (r,c,f) and edges are moves.
        # Use 0-1 BFS.
        # But we can't build entire graph.
        
        # Let's think differently: 
        # For each query, we start at (A,B,Y) and want to reach (C,D,Z).
        # We perform 0-1 BFS from (A,B,Y).
        # We only expand to (r,c,f) such that 1 <= f <= F[r][c].
        # This way, we don't precompute everything.
        
        # visited dict: (r,c,f) -> cost
        from collections import deque
        visited = {}
        q = deque()
        q.append((0, s_r, s_c, s_f))
        visited[(s_r, s_c, s_f)] = 0
        
        while q:
            cost, r, c, f = q.popleft()
            if cost > visited.get((r,c,f), float('inf')):
                continue
                
            if (r,c,f) == (t_r, t_c, t_f):
                return cost
            
            # Move up/down stairs (within same building)
            if f > 1:
                nf = f - 1
                if (r,c,nf) not in visited or visited[(r,c,nf)] > cost + 1:
                    visited[(r,c,nf)] = cost + 1
                    q.append((cost + 1, r, c, nf))
            if f < F[r][c]:
                nf = f + 1
                if (r,c,nf) not in visited or visited[(r,c,nf)] > cost + 1:
                    visited[(r,c,nf)] = cost + 1
                    q.append((cost + 1, r, c, nf))
            
            # Walkway moves to adjacent blocks (same floor)
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    if F[nr][nc] >= f:
                        if (nr, nc, f) not in visited or visited[(nr, nc, f)] > cost:
                            visited[(nr, nc, f)] = cost
                            q.appendleft((cost, nr, nc, f))
        
        return -1  # Should not happen per problem statement
    
    results = []
    for A, B, Y, C, D, Z in queries:
        res = bfs_01(A, B, Y, C, D, Z)
        results.append(str(res))
    
    print('\n'.join(results))

if __name__ == "__main__":
    main()
