import sys

def main():
    data = sys.stdin.read().strip().split()
    if len(data) != 2:
        return
    s1, s2 = data
    # Map strings to boolean: True means sick
    t_sick = (s1 == "sick")
    a_sick = (s2 == "sick")
    for x in range(1, 5):
        t_has = x in (1, 2)
        a_has = x in (1, 3)
        if (t_has == t_sick) and (a_has == a_sick):
            print(x)
            return

if __name__ == "__main__":
    main()
