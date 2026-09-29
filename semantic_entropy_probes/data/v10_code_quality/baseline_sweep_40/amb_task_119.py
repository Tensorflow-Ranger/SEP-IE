import sys

def main():
    S = sys.stdin.readline().strip()
    result = []
    i = 0
    while i < len(S) - 1:
        if S[i] == 'W' and S[i+1] == 'A':
            result.append('AC')
            i += 2
        else:
            result.append(S[i])
            i += 1
    print(''.join(result))

if __name__ == "__main__":
    main()
