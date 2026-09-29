import sys

def main():
    N, M = map(int, sys.stdin.read().split())
    if N == 1:
        print(M + 1)
    else:
        if M == 0:
            print(1)
        else:
            total = (N ** (M + 1) - 1) // (N - 1)
            print(total if N != 1 else M + 1)

if __name__ == "__main__":
    main()
