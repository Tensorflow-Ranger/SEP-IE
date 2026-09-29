```python
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # We can think of this problem in terms of linear algebra over GF(2).
    # Each stone can be viewed as a vector in GF(2)^60 (since numbers go up to 10^17 < 2^60).
    # The operations allowed are moving all stones from one bag to another,
    # which corresponds to adding the vector of that bag to another.
    # This is equivalent to being able to form any subset sum of the original vectors
    # by choosing which bags to keep non-empty (with some coefficient corresponding to how many times we add).
    # But since we can move entire bags, we can actually achieve any subset XOR of the original A_i's.
    # That is, we can choose any non-empty subset of the A_i's and XOR them together.
    # However, there's also the constraint that at least one bag must remain (to have a XOR),
    # but since we start with N>=2, and we can always consolidate into fewer bags,
    # we can get any XOR of any non-empty subset of the original A_i's.
    # Actually, let's re-think:
    # - We start with bags having A_1,...,A_N stones.
    # - We can move all stones from one bag to another.
    # - Goal: Final XOR of all bags' stone counts.
    # Key insight: We can consolidate any subset of the initial bags into a single bag.
    # So the final configuration can have any subset of the original A_i's XORed together,
    # depending on which subsets we merge into individual bags.
    # More precisely, we can partition the set {A_1,...,A_N} into non-empty groups,
    # and each group will contribute the XOR of its elements to the final result.
    # Then the overall result is the XOR of these group contributions.
    # But wait, we can do better. Since we can move all stones from one bag to another,
    # we can effectively choose which A_i's to XOR together.
    # For example, if we want A_1 XOR A_2 XOR A_4, we can:
    # - Move all from bag 3 to bag 1 (now bag 1 has A_1+A_3)
    # - Move all from bag 5 to bag 2 (now bag 2 has A_2+A_5)
    # But that doesn't directly give us XOR.
    # Let's think differently.
    # The key observation is that we can achieve any XOR of a non-empty subset of the A_i's.
    # Because:
    # - We can choose to keep any subset of the original bags non-empty by moving others into them.
    # - If we want just A_i, we can move all other stones into one other bag, leaving A_i alone.
    # - If we want A_i XOR A_j, we can move all stones from bag j into bag i, so bag i has A_i+A_j stones,
    #   but that's addition, not XOR.
    # Hmm, perhaps I'm overcomplicating.
    # Let's consider what operations we can do:
    # Operation: Choose two bags A and B, move all stones from A to B.
    # So if bag A has x stones, bag B has y stones, after operation:
    # Bag A has 0 stones, bag B has x+y stones.
    # We want to find possible values of XOR of final stone counts.
    # Let’s think of small examples:
    # N=2: A=[a,b]
    # Initial: [a,b] -> XOR = a XOR b
    # Move all from bag 1 to bag 2: [0, a+b] -> XOR = 0 XOR (a+b) = a+b
    # Move all from bag 2 to bag 1: [a+b, 0] -> XOR = a+b XOR 0 = a+b
    # So possible values are {a XOR b, a+b}. Answer = 2.
    # But wait, maybe we can do more moves?
    # From [0, a+b], we can't do anything else (bag 1 is empty).
    # From [a+b, 0], same thing.
    # So for N=2, answer is number of distinct values among:
    # - a XOR b
    # - a+b
    # So at most 2 values.
    # But actually, we can also get a or b by moving the other away.
    # Wait no, once we move all from one bag, that bag stays empty.
    # So for N=2, starting [a,b]:
    # - Do nothing: XOR = a XOR b
    # - Move 1->2: [0, a+b] => XOR = a+b
    # - Move 2->1: [a+b, 0] => XOR = a+b
    # So possible values are {a XOR b, a+b}.
    # What about N=3? A=[a,b,c].
    # We can get:
    # - a XOR b XOR c (do nothing)
    # - Move 1->2: [0, b+a, c] => XOR = (a+b) XOR c
    # - Move 1->3: [0, b, c+a] => XOR = b XOR (c+a)
    # - Move 2->1: [a+b, 0, c] => XOR = (a+b) XOR c
    # - Move 2->3: [a, 0, c+b] => XOR = a XOR (c+b)
    # - Move 3->1: [a+c, b, 0] => XOR = (a+c) XOR b
    # - Move 3->2: [a, b+c, 0] => XOR = a XOR (b+c)
    # And then we can continue from these states.
    # For example from [0, b+a, c]:
    # - Move 2->3: [0, 0, c+b+a] => XOR = a+b+c
    # - Move 3->2: [0, c+b+a, 0] => XOR = a+b+c
    # From [0, b, c+a]:
    # - Move 2->1: [b, 0, c+a] => XOR = b XOR (c+a)
    # - Move 2->3: [0, 0, c+a+b] => XOR = a+b+c
    # - Move 3->1: [c+a, b, 0] => XOR = (c+a) XOR b
    # - Move 3->2: [0, b+c+a, 0] => XOR = a+b+c
    # It seems like we can get various subset XORs.
    # In fact, the key insight is that we can achieve any XOR of a non-empty subset of the original A_i's.
    # Because we can move stones around to isolate any subset.
    # For example, to get just A_i, we can move all other A_j (j != i) into one specific other bag,
    # leaving A_i alone in its bag.
    # To get A_i XOR A_j, we can move all other A_k (k != i
