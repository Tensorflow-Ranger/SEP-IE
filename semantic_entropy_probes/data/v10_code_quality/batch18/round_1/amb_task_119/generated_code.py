import sys

def main():
    import sys
    input = sys.stdin.read
    S = input().strip()
    S = list(S)
    i = 0
    while i < len(S) - 1:
        if S[i] == 'W' and S[i+1] == 'A':
            S[i] = 'A'
            S[i+1] = 'C'
            i = 0
        else:
            i += 1
    print(''.join(S))

if __name__ == "__main__":
    main()
