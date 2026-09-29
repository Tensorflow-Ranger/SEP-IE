import sys

def main():
    N, M = map(int, sys.stdin.read().split())
    if N == 1:
        print(M + 1)
    else:
        if M == -1:
            print("inf")
        else:
            total = (pow(N, M + 1) - 1) // (N - 1)
            print(total)

if __name__ == "__main__":
    main()
