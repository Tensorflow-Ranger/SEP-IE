import sys
from collections import defaultdict

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    
    N, M = map(int, sys.stdin.readline().split())
    
    if M == 0:
        # No constraints, all zeros is optimal
        print(' '.join(['0'] * N))
        return

    # We'll model this as a system of XOR equations:
    # For each constraint (x, y, z): A[x] ^ A[y] = z
    # This can be modeled as a graph where each component is a connected component.
    # Within each component, fixing one variable determines all others.
    # We can use Union-Find with a twist: for each node, we store its XOR relative to the root.

    parent = list(range(N + 1))
    xor_val = [0] * (N + 1)  # xor_val[i] = A[i] ^ A[root[i]]

    def find(x):
        if parent[x] != x:
            old_parent = parent[x]
            parent[x] = find(parent[x])
            xor_val[x] ^= xor_val[old_parent]
        return parent[x]

    def union(x, y, z):
        # We want A[x] ^ A[y] = z
        # Let rx = root of x, ry = root of y
        # A[x] ^ A[rx] = xor_val[x]
        # A[y] ^ A[ry] = xor_val[y]
        # A[x] ^ A[y] = z
        # => (A[rx] ^ xor_val[x]) ^ (A[ry] ^ xor_val[y]) = z
        # => A[rx] ^ A[ry] = xor_val[x] ^ xor_val[y] ^ z
        rx = find(x)
        ry = find(y)
        if rx == ry:
            # Check consistency
            if (xor_val[x] ^ xor_val[y]) != z:
                return False
            return True
        else:
            # Merge ry into rx
            # A[rx] ^ A[ry] = xor_val[x] ^ xor_val[y] ^ z
            # So we set parent[ry] = rx and xor_val[ry] = xor_val[x] ^ xor_val[y] ^ z
            parent[ry] = rx
            xor_val[ry] = xor_val[x] ^ xor_val[y] ^ z
            return True

    # Process all constraints
    for _ in range(M):
        x, y, z = map(int, sys.stdin.readline().split())
        if not union(x, y, z):
            print(-1)
            return

    # Now, for each connected component, we can choose the root value to minimize sum.
    # For each component, collect all nodes and their xor relative to root.
    components = defaultdict(list)
    for i in range(1, N + 1):
        root = find(i)
        components[root].append((i, xor_val[i]))

    A = [0] * (N + 1)

    # For each component, we want to minimize sum of A[i] = root_val ^ xor_val[i]
    # This is sum over i of (root_val ^ xor_val[i])
    # To minimize this, for each bit position, we choose root_val's bit to minimize the count of 1s in the result.
    for root, nodes in components.items():
        # nodes is list of (index, xor_wrt_root)
        # We want to choose root_val to minimize sum of (root_val ^ xor_val)
        # Let's consider each bit separately.
        # But since values can be up to 10^9, we can't do bit-by-bit easily.
        # Actually, for a fixed root_val, the sum is sum_i (root_val ^ xor_val[i])
        # To minimize this sum, we can compute for root_val = 0 and root_val = some small values,
        # but that's not efficient.
        # Key insight: we can choose root_val = 0, which gives sum = sum of xor_val[i].
        # But we want to minimize the sum. 
        # For each bit k, count how many xor_val[i] have bit k set.
        # If more than half have it set, flipping that bit in root_val reduces the sum.
        # But we need to be careful about the range.
        # Actually, the optimal root_val is such that for each bit, 
        # if more than half of the xor_vals have that bit set, we set that bit in root_val to 0,
        # otherwise to 1. Wait no, if more have it set, setting root_val bit to 1 will make
        # root_val ^ xor_val have that bit 0 for those, and 1 for others.
        # Let's think: for bit k, let c be count of xor_vals with bit k set.
        # If root_val has bit k = 0: we get c ones => contributes c * 2^k to sum
        # If root_val has bit k = 1: we get (total - c) ones => contributes (total - c) * 2^k to sum
        # So choose bit k of root_val to be 1 if (total - c) < c, i.e., if c > total/2
        # No, that's wrong. If root_val bit is 1, then:
        # xor_val bit 1 => result bit 0
        # xor_val bit 0 => result bit 1
        # So we get (n-c) ones.
        # If root_val bit is 0:
        # xor_val bit 1 => result bit 1
        # xor_val bit 0 => result bit 0
        # So we get c ones.
        # We want to minimize, so if c <= n-c, choose bit=0, else choose bit=1.
        # That is, choose bit=0 if c <= n/2, else bit=1. If n is even and c=n/2, either is fine.
        
        # But since xor_vals can be up to 10^9, we need to consider up to 30 bits.
        # Let's compute optimal root_val for this component.
        if not nodes:
            continue
        n = len(nodes)
        xor_vals = [x[1] for x in nodes]
        
        # Determine optimal root_val for this component
        root_val = 0
        max_bit = 31
        for bit in range(max_bit):
            count = 0
            mask = 1 << bit
            for val in xor_vals:
                if val & mask:
                    count += 1
            # If count > n - count, better to set root's bit to 1
            if count > n - count:
                root_val |= mask
        
        # Assign values
        for idx, xor_wrt_root in nodes:
            A[idx] = root_val ^ xor_wrt_root

    print(' '.join(map(str, A[1:])))

if __name__ == "__main__":
    main()
