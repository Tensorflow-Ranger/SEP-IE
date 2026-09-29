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
    
    # Precompute for each floor level the reachable cells
    max_floor = 0
    for i in range(H):
        for j in range(W):
            max_floor = max(max_floor, F[i][j])
    
    # For each floor level x, build a grid of whether cell (i,j) has >= x floors
    # We'll use 0-based indexing for floors: floor 1 -> index 0, ..., floor f -> index f-1
    # But since floors are numbered from 1, we consider floors 1 to max_floor
    # To avoid confusion, we'll work directly with floor numbers (1-based)
    
    # Instead of precomputing all, we'll do BFS per query with memoization
    # But since H,W <=500 and Q<=10^5, we need efficient approach
    
    # Let's think differently: model this as a graph where state is (row, col, floor)
    # Transitions:
    # 1. Move up/down within building: cost 1 per floor
    # 2. Move to adjacent building with same floor: cost 0
    
    # Since movement between buildings is cost 0, we can use 0-1 BFS
    # State: (row, col, floor)
    # But floors can be up to 1e6 => too many states
    
    # Key insight: we only care about floors that exist in at least one building
    # Even better: for each query, we can limit search to relevant floors
    
    # Alternative: for each query, run 0-1 BFS on the state graph
    # But again, floors can be large
    
    # Better idea: 
    # Model the state space as (row, col, floor)
    # Use 0-1 BFS where:
    # - Moving between same floor in adjacent buildings: cost 0
    # - Moving up/down one floor in same building: cost 1
    
    # However, with floors up to 1e6, even 0-1 BFS might be too slow unless optimized
    
    # Optimization: compress the floor space
    # For each query, collect all floors in the buildings involved and compress
    
    results = []
    
    for A, B, Y, C, D, Z in queries:
        # Collect all floors in all buildings for coordinate compression
        floors_set = set()
        floors_set.add(Y)
        floors_set.add(Z)
        for i in range(H):
            for j in range(W):
                floors_set.add(F[i][j])
        
        floors_list = sorted(floors_set)
        floor_to_idx = {f: i for i, f in enumerate(floors_list)}
        num_floors = len(floors_list)
        idx_to_floor = floors_list
        
        # Now perform 0-1 BFS on states (row, col, compressed_floor_index)
        # But still too many states potentially
        
        # Even better optimization:
        # Only consider states that are reachable
        # Run 0-1 BFS but limit to necessary floors
        
        # Actually, let's reconsider the constraints:
        # H,W <= 500, Q <= 10^5
        # So we can't do full preprocessing
        # Need to solve each query independently efficiently
        
        # Let's try modeling with 0-1 BFS properly
        # State: (row, col, floor)
        # But we need to limit the number of states
        
        # Idea: for each query, we can limit to floors that appear in any building
        # And also limit spatially? No, because walkways can go anywhere
        
        # Maybe we don't compress floors, but instead use actual floor numbers
        # and limit the BFS appropriately
        
        # Let's proceed with 0-1 BFS on (row, col, floor) with actual floor numbers
        # But prune states smartly
        
        from collections import deque
        
        # dist[(r,c,f)] = minimum stair steps to reach floor f in building (r,c)
        dist = {}
        queue = deque()
        
        start_state = (A, B, Y)
        dist[start_state] = 0
        queue.append(start_state)
        
        while queue:
            r, c, f = queue.popleft()
            current_dist = dist[(r, c, f)]
            
            # If we reached the target
            if r == C and c == D and f == Z:
                results.append(current_dist)
                break
            
            # Move up/down within building
            if f > 1:
                new_state = (r, c, f - 1)
                if new_state not in dist or dist[new_state] > current_dist + 1:
                    dist[new_state] = current_dist + 1
                    queue.append(new_state)
            
            if f < F[r][c]:
                new_state = (r, c, f + 1)
                if new_state not in dist or dist[new_state] > current_dist + 1:
                    dist[new_state] = current_dist + 1
                    queue.append(new_state)
            
            # Move to adjacent buildings with same floor
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    if F[nr][nc] >= f:
                        new_state = (nr, nc, f)
                        if new_state not in dist or dist[new_state] > current_dist:
                            dist[new_state] = current_dist
                            queue.appendleft(new_state)
        else:
            # This shouldn't happen as per problem statement
            results.append(-1)
    
    for res in results:
        print(res)

if __name__ == "__main__":
    main()
