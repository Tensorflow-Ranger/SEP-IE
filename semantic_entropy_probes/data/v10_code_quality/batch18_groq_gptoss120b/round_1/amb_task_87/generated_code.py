import sys

def main():
    data = sys.stdin.read().strip().split()
    if len(data) != 2:
        return
    S1, S2 = data
    # Map strings to boolean: True means sick
    sick_T = (S1 == "sick")
    sick_A = (S2 == "sick")
    for x in range(1, 5):
        takahashi_sick = (x == 1 or x == 2)
        aoki_sick = (x == 1 or x == 3)
        if takahashi_sick == sick_T and aoki_sick == sick_A:
            print(x)
            return

if __name__ == "__main__":
    main()
