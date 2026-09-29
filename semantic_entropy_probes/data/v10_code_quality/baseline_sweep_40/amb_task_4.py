import sys
import itertools

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    total = sum(A)
    
    # We can redistribute stones, so the total remains the same
    # The XOR of all B_i is the same as the XOR of all A_i, because the total number of stones is fixed
    # However, we can choose which bag has how many stones, as long as the total is fixed
    # The key is to find all possible XOR values of N non-negative integers that sum to total
    
    # We can try all possible ways to split the total into N non-negative integers
    # But since N is small (<=12), we can use a recursive approach or bitmasking
    
    # We can use a set to store all possible XOR values
    possible_xors = set()
    
    # We use a recursive function to generate all possible distributions of stones
    def dfs(pos, current_sum, current_xor, path):
        if pos == N:
            possible_xors.add(current_xor)
            return
        # Try all possible values for the current position
        # The value can be from 0 to total - current_sum
        max_val = total - current_sum
        for val in range(0, max_val + 1):
            dfs(pos + 1, current_sum + val, current_xor ^ val, path + [val])
    
    dfs(0, 0, 0, [])
    
    print(len(possible_xors))

if __name__ == "__main__":
    main()
