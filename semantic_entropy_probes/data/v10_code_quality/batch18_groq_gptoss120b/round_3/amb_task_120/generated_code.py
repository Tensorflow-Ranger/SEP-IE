S = input().strip()

while True:
    pos = S.find('WA')
    if pos == -1:
        break
    S = S[:pos] + 'AC' + S[pos+2:]

print(S)
