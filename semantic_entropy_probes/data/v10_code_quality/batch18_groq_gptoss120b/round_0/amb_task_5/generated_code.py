import sys

sys.setrecursionlimit(10000)

def main() -> None:
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    A = list(map(int, data[1:]))

    answers = set()

    def dfs(pos: int, groups: list):
        if pos == N:
            x = 0
            for s in groups:
                x ^= s
            answers.add(x)
            return
        val = A[pos]
        # add to existing groups
        for i in range(len(groups)):
            groups[i] += val
            dfs(pos + 1, groups)
            groups[i] -= val
        # start new group
        groups.append(val)
        dfs(pos + 1, groups)
        groups.pop()

    dfs(0, [])
    print(len(answers))

if __name__ == "__main__":
    main()
