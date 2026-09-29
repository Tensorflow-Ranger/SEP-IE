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
    
    # The XOR of all final B_i is equal to the XOR of all A_i, because moving stones doesn't change the total XOR
    # So the answer is 1 if all B_i can be made such that their XOR is total_xor
    # But since we can merge any number of bags into one, the only possible XOR is total_xor
    # Because the XOR of all B_i is fixed as the XOR of all A_i
    print(1)

if __name__ == "__main__":
    main()
