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
    black = []
    white = []
    
    for b in B:
        black.append(b)
        if len(black) > len(white):
            white.append(heapq.heappop(black))
    
    for w in W:
        if len(white) < len(black):
            heapq.heappush(white, w)
        else:
            if w > white[0]:
                heapq.heappushpop(white, w)
    
    total = sum(black) + sum(white)
    print(total)

if __name__ == "__main__":
    main()
