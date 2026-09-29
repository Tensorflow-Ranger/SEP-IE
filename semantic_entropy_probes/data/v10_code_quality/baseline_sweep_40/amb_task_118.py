import sys

def replace_wa(s):
    s = list(s)
    i = 0
    while i <= len(s) - 2:
        if s[i] == 'W' and s[i+1] == 'A':
            s[i] = 'A'
            s[i+1] = 'C'
            i += 2
        else:
            i += 1
    return ''.join(s)

def main():
    s = sys.stdin.readline().strip()
    print(replace_wa(s))

if __name__ == "__main__":
    main()
