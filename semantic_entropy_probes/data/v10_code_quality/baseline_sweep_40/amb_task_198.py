import sys
import heapq

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    B = list(map(int, data[2:2+N]))
    W = list(map(int, data[2+N:2+N+M]))
    
    B.sort(reverse=True)
    W.sort(reverse=True)
    
    max_sum = 0
    total = 0
    i = 0
    j = 0
    
    while i < N and j < M:
        if B[i] > W[j]:
            total += B[i]
            i += 1
        else:
            total += W[j]
            j += 1
    
    # Now, we have the maximum possible sum for the minimum of N and M pairs
    # But we need to ensure that the number of black and white balls chosen is equal
    # So we need to adjust the total by removing the smallest elements from the combined list
    
    # Create a list of the first min(N, M) elements from both sorted lists
    combined = []
    for k in range(min(N, M)):
        combined.append(B[k])
        combined.append(W[k])
    
    # Sort the combined list in ascending order
    combined.sort()
    
    # Remove the smallest element to make the counts equal
    if len(combined) > 0:
        total -= combined[0]
    
    print(total)

if __name__ == "__main__":
    main()
