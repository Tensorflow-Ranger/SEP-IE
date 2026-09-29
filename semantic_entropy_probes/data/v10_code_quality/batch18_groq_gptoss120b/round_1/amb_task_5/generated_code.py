import sys

def solve() -> None:
    data = sys.stdin.read().strip().split()
    if not data:
        return
    n = int(data[0])
    A = list(map(int, data[1:]))

    results = set()

    def dfs(idx, groups):
        if idx == n:
            x = 0
            for g in groups:
                x ^= g
            results.add(x)
            return
        val = A[idx]
        # add to existing groups
        for i in range(len(groups)):
            groups[i] += val
            dfs(idx + 1, groups)
            groups[i] -= val
        # start new group
        groups.append(val)
        dfs(idx + 1, groups)
        groups.pop()

    dfs(0, [])
    print(len(results))

if __name__ == "__main__":
    solve()
