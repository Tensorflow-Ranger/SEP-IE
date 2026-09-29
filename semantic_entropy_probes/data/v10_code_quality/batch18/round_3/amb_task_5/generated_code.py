import sys
import itertools

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    total_xor = 0
    for a in A:
        total_xor ^= a
    
    # The XOR of all B_i is equal to the XOR of all A_i, because moving stones doesn't change the total XOR
    # So the only possible value is total_xor
    # But wait, this is not correct. Because we can merge bags, which changes the individual B_i values, but the total XOR remains the same.
    # However, the XOR of all B_i is equal to the XOR of all A_i, because the total number of stones is the same.
    # So the XOR of all B_i is fixed and equal to the XOR of all A_i.
    # Therefore, the answer is 1.
    print(1)

if __name__ == "__main__":
    main()
