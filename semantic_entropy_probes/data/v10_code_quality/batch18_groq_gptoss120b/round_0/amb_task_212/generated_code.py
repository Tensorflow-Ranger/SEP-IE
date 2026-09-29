ru, pu = find(u)          # pu = value[u] xor value[ru]
rv, pv = find(v)

if ru == rv:
        # already in same component, check consistency
        if (pu xor pv) != w: contradiction
else:
        # attach ru under rv
        parent[ru] = rv
        # we need parity[ru] so that the equation holds:
        # (pu xor parity[ru]) xor pv = w   => parity[ru] = pu xor pv xor w
        parity[ru] = pu xor pv xor w
