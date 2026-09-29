"""v2 of the redundant-vs-missing-detail probe test
(scripts/build_detail_addition_dataset.py), fixing a leakage problem found
by inspection: v1's 10 base prompts were mostly in the *training* split of
the probes being evaluated (6/10 in both), so the baseline predictions were
partly recalled rather than generalized.

This version selects two separate, disjoint sets of 10 K=1 ClarifyCodeBench
prompts, each restricted to source problems that landed in the TEST split of
the specific probe(s) being evaluated against (reconstructed via the same
grouped-split code + seed used at training time -- see
scripts/check_detail_addition_splits.py):

  - `mc` set (tag=multiclass): 10 prompts, all in results_oatml_multiclass's
    test split (activations/, K>=1, src/splits.make_train_val_test_split,
    seed=0).
  - `lrmlp` set (tag=binary): 10 prompts, all in models/rank{1,2}_*'s test
    split (activations_ccb_heval/, src/combined_splits.
    make_combined_train_val_test_split, seed=0) -- both rank1 (LR) and
    rank2 (MLP) share this exact split, confirmed identical.

Same construction as v1 otherwise: for each base (K=1, single ambiguity)
prompt, one "redundant" bullet (restates something already stated/implied)
and one "missing" bullet (ClarifyCodeBench's own annotated clarification,
reworded to read as a natural Constraints bullet, not a literal
"Clarifications:" header -- avoiding FINDINGS.md's Confound 1).
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "detail_addition_probe_test_v2.jsonl"

# tag: "multiclass" -> held out for results_oatml_multiclass/probe.pkl only
#      "binary"     -> held out for both models/rank1_lr*.joblib and rank2_mlp*.joblib
ITEMS = [
    # ---------------- multiclass-probe test-split set ----------------
    dict(
        tag="multiclass", base_id="amb_task_11", source_problem_id="abc390_f",
        ambiguity_types=["Terminology"],
        prompt=(
            "You are given an integer sequence A=(A_1,A_2,\\ldots,A_N) of length N.\n"
            "For each integer pair (L,R) with 1 \\le L \\le R \\le N, define f(L,R) as follows:\n"
            "- Start with an empty blackboard. Write the R-L+1 integers A_L, A_{L+1}, \\ldots, A_R on the blackboard.\n"
            "- Repeat the following operation until all integers on the blackboard are erased:\n"
            "- Choose integers l, r with l \\le r such that every integer from l through r appears at least once "
            "on the blackboard. Then, erase all integers from l through r that are on the blackboard.\n"
            "- Let f(L,R) be the minimum number of such operations needed to erase all the integers from the blackboard.\n"
            "Find \\displaystyle \\sum_{L=1}^N \\sum_{R=L}^N f(L,R).\n"
            "Input\nThe input is given from Standard Input in the following format:\nN\nA_1 A_2 \\ldots A_N\n"
            "Output\nPrint the answer.\n"
            "Constraints\n- 1 \\le N \\le 3 \\times 10^5\n- 1 \\le A_i \\le N\n- All input values are integers."
        ),
        redundant_bullet="N and the sequence A_1,\\ldots,A_N are given via standard input as shown in the input format above.",
        missing_bullet="The integers A_L, A_{L+1}, \\ldots, A_R are written on the blackboard in their original order; the order matters.",
    ),
    dict(
        tag="multiclass", base_id="amb_task_12", source_problem_id="abc390_c",
        ambiguity_types=["Terminology"],
        prompt=(
            "You are given a grid of H rows and W columns.\n"
            "Let (i,j) denote the cell at row i (1 \\leq i \\leq H) from the top and column j (1 \\leq j \\leq W) from the left.\n"
            "The state of the grid is represented by H strings S_1, S_2, \\ldots, S_H, each of length W. \n"
            "Takahashi wants to paint each not-yet-painted cell so that all the black cells form a rectangle.\n"
            "More precisely, he wants there to exist a quadruple of integers (a,b,c,d) (1 \\leq a \\leq b \\leq H, "
            "1 \\leq c \\leq d \\leq W) such that:\n"
            "For each cell (i,j) (1 \\leq i \\leq H, 1 \\leq j \\leq W),\n"
            "if a \\leq i \\leq b and c \\leq j \\leq d, the cell is black;\n"
            "otherwise, the cell is white.\n"
            "Determine whether this is possible.\n"
            "Input\nThe input is given from Standard Input in the following format:\nH W\nS_1\nS_2\n\\vdots\nS_H\n"
            "Output\nIf it is possible to paint all the not-yet-painted cells so that the black cells form a rectangle, "
            "print Yes; otherwise, print No.\n"
            "Constraints\n- 1 \\leq H, W \\leq 1000\n- H and W are integers.\n"
            "- Each S_i is a string of length W consisting of #, ., ?.\n"
            "- There is at least one cell that is already painted black."
        ),
        redundant_bullet="H and W are given on the first line, followed by the H strings S_1,\\ldots,S_H, as shown in the input format above.",
        missing_bullet="In each S_i, # denotes a black cell, . denotes a white cell, and ? denotes a cell not yet painted.",
    ),
    dict(
        tag="multiclass", base_id="amb_task_108", source_problem_id="abc393_d",
        ambiguity_types=["Indices & Ranges"],
        prompt=(
            "You are given a string S of length N consisting of 0 and 1. It is guaranteed that S contains at least one 1.\n"
            "You may perform the following operation any number of times (possibly zero):\n"
            "- Choose an integer i and swap the i-th and (i+1)-th characters of S.\n"
            "Find the minimum number of operations needed so that all 1s are contiguous.\n"
            "Here, all 1s are said to be contiguous if and only if there exist integers l and r (1 \\leq l \\leq r \\leq N) "
            "such that the i-th character of S is 1 if and only if l \\leq i \\leq r, and 0 otherwise.\n"
            "Input\nThe input is given from Standard Input in the following format:\nN\nS\n"
            "Output\nPrint the answer.\n"
            "Constraints\n- 2 \\leq N \\leq 5 \\times 10^5\n- N is an integer.\n"
            "- S is a length N string of 0 and 1.\n- S contains at least one 1."
        ),
        redundant_bullet="S is given as a length-N string on the line following N, as shown in the input format above.",
        missing_bullet="The index i must satisfy 1 \\leq i \\leq N-1, so only adjacent characters within the string may be swapped.",
    ),
    dict(
        tag="multiclass", base_id="amb_task_177", source_problem_id="abc395_a",
        ambiguity_types=["Terminology"],
        prompt=(
            "You are given a positive integer N and a sequence of positive integers A = (A_1,A_2,\\dots,A_N) of length N.\n"
            "Determine whether A is strictly increasing.\n"
            "Input\nThe input is given from Standard Input in the following format:\nN\nA_1 A_2 \\dots A_N\n"
            "Output\nIf A is strictly increasing, print Yes; otherwise, print No.\n"
            "The judge is case-insensitive. For example, if the correct answer is Yes, any of yes, YES, and yEs will be accepted.\n"
            "Constraints\n- 2 \\leq N \\leq 100\n- 1 \\leq A_i \\leq 1000 \\ (1 \\leq i \\leq N)\n- All input values are integers."
        ),
        redundant_bullet="N is given on the first line, followed by A_1,\\ldots,A_N on the second line, as shown above.",
        missing_bullet="“Strictly increasing” means A_i < A_{i+1} must hold for every i with 1 \\leq i < N.",
    ),
    dict(
        tag="multiclass", base_id="amb_task_205", source_problem_id="abc396_b",
        ambiguity_types=["Collection Semantics"],
        prompt=(
            "There is a stack of 100 cards.\n"
            "Process Q queries. Each query is of one of the following:\n"
            "- Type 1: Place a card labeled with an integer x on top of the stack.\n"
            "- Type 2: Remove the top card of the stack and output the integer written on that removed card. "
            "Under the constraints of this problem, the stack always has at least one card.\n"
            "Input\nThe input is given from Standard Input in the following format:\n"
            "Q\n\\text{query}_1\n\\text{query}_2\n\\vdots\n\\text{query}_Q\n"
            "The i-th query \\text{query}_i starts with the query type c_i (1 or 2), followed by the integer x if c_i=1.\n"
            "That is, each query is in one of the following two formats:\n1 x\n2\n"
            "Output\nLet q be the number of queries with c_i=2. Print q lines.\n"
            "The j-th line (1 \\le j \\le q) should contain the answer to the j-th such query.\n"
            "Constraints\n- 1 \\le Q \\le 100\n- 1 \\le x \\le 100\n"
            "- There is at least one query of type 2.\n- All input values are integers."
        ),
        redundant_bullet="Each query is given on its own line in the format shown above, starting with the query type c_i.",
        missing_bullet="Each of the 100 initial cards in the stack is labeled with the integer 0.",
    ),
    dict(
        tag="multiclass", base_id="amb_task_243", source_problem_id="abc397_d",
        ambiguity_types=["Output Format"],
        prompt=(
            "You are given a positive integer N. Determine whether there exists a pair of positive integers (x,y) "
            "such that x^3 - y^3 = N. If such a pair exists, print one such pair (x,y).\n"
            "Input\nThe input is given from Standard Input in the following format:\nN\n"
            "Output\nIf there is such a pair, print x and y in this order separated by a space. "
            "If there are multiple solutions, printing any one of them is accepted as correct.\n"
            "Constraints\n- 1 \\leq N \\leq 10^{18}\n- All input values are integers."
        ),
        redundant_bullet="N is given as a single integer on standard input, as shown in the input format above.",
        missing_bullet="If no such pair (x,y) exists, print -1.",
    ),
    dict(
        tag="multiclass", base_id="amb_task_274", source_problem_id="abc398_f",
        ambiguity_types=["Behavior"],
        prompt=(
            "Find one palindrome that has S as its prefix.\n"
            "Input\nThe input is given from Standard Input in the following format:\nS\n"
            "Output\nPrint the answer.\nIf multiple solutions exist, any of them is accepted.\n"
            "Constraints\n- S is a string of length between 1 and 500000, inclusive, consisting of uppercase English letters."
        ),
        redundant_bullet="S is given as a single line of standard input, as shown in the input format above.",
        missing_bullet="Among all palindromes having S as a prefix, output the shortest one.",
    ),
    dict(
        tag="multiclass", base_id="amb_task_305", source_problem_id="abc399_c",
        ambiguity_types=["Behavior"],
        prompt=(
            "You are given a simple undirected graph with N vertices and M edges, where the vertices are labeled 1 to N. "
            "The i-th edge connects vertices u_i and v_i.\n"
            "What is the number of edges that need to be deleted from this graph so that the graph becomes a forest?\n"
            "What is a forest?\nA simple undirected graph F is called a forest if and only if F does not contain any cycle.\n"
            "Input\nThe input is given from Standard Input in the following format:\nN M\nu_1 v_1\nu_2 v_2\n\\vdots\nu_M v_M\n"
            "Output\nPrint the answer.\n"
            "Constraints\n- 1 \\leq N \\leq 2 \\times 10^5\n"
            "- 0 \\leq M \\leq \\min \\left( \\frac{N(N-1)}{2}, 2 \\times 10^5\\right)\n"
            "- 1 \\leq u_i < v_i \\leq N\n- The given graph is simple.\n- All input values are integers."
        ),
        redundant_bullet="N and M are given on the first line, followed by the M edges, as shown in the input format above.",
        missing_bullet="Find the minimum number of edges that must be deleted to make the graph a forest.",
    ),
    dict(
        tag="multiclass", base_id="amb_task_309", source_problem_id="abc399_b",
        ambiguity_types=["Edge Cases"],
        prompt=(
            "N people labeled from 1 to N participated in a certain contest. The score of person i (1 \\leq i \\leq N) was P_i.\n"
            "In this contest, the rank of each of the N people is determined by the following procedure:\n"
            "- Prepare a variable r, and initialize r. Initially, the ranks of the N people are all undetermined.\n"
            "- Repeat the following operation until the ranks of all N people are determined:\n"
            "- Let x be the maximum score among the people whose ranks are currently undetermined, and let k be the "
            "number of people whose score is x. Determine the rank of those k people with score x to be r, and then add k to r.\n"
            "Print the rank of each of the N people.\n"
            "Input\nThe input is given from Standard Input in the following format:\nN\nP_1 P_2 \\dots P_N\n"
            "Output\nPrint N lines. The i-th line (1 \\leq i \\leq N) should contain the rank of person i as an integer.\n"
            "Constraints\n- 1\\leq N \\leq 100\n- 1\\leq P_i \\leq 100\n- All input values are integers."
        ),
        redundant_bullet="N is given on the first line, followed by P_1,\\ldots,P_N on the second line, as shown above.",
        missing_bullet="The variable r is initialized to 1, so the first group of highest-scoring people receives rank 1.",
    ),
    dict(
        tag="multiclass", base_id="amb_task_142", source_problem_id="abc394_g",
        ambiguity_types=["Indices & Ranges"],
        prompt=(
            "There is a city divided into H \\times W blocks, and there is exactly one building in each block.\n"
            "Specifically, in the block at the i-th row from the north (1\\leq i\\leq H) and the j-th column from the "
            "west (1\\leq j\\leq W) (hereafter referred to as block (i,j)), there is a building of F_{i,j} floors.\n"
            "Takahashi has two ways of moving. If he is on the X-th floor (1\\leq X\\leq F_{i,j}) of the building in "
            "block (i,j), he can:\n"
            "- Move up or down one floor within the same building using stairs. If X=1, he cannot move down; "
            "if X=F_{i,j}, he cannot move up.\n"
            "- Choose a building with at least X floors in a cardinally adjacent block, and move to the X-th floor of "
            "that building using a (sky) walkway.\n"
            "Here, two blocks (i,j) and (i',j') are cardinally adjacent if and only if \\lvert i - i'\\rvert + "
            "\\lvert j - j'\\rvert = 1.\n"
            "You are given Q queries to be answered. The i-th query (1\\leq i\\leq Q) is the following.\n"
            "Find the minimum possible number of times that Takahashi uses stairs to move from the Y_i-th floor of "
            "the building in block (A_i,B_i) to the Z_i-th floor of the building in block (C_i,D_i).\n"
            "The count of times using stairs is incremented each time he moves up or down one floor, possibly multiple "
            "times within the same building. (For example, moving from the 1st floor to the 6th floor of a building "
            "counts as 5 uses of stairs.)\n"
            "Note that he does not have to minimize the number of times he uses walkways.\n"
            "Input\nThe input is given from Standard Input in the following format:\n"
            "H W\nF_{1,1} F_{1,2} \\ldots F_{1,W}\nF_{2,1} F_{2,2} \\ldots F_{2,W}\n\\vdots\nF_{H,1} F_{H,2} \\ldots F_{H,W}\n"
            "Q\nA_1 B_1 Y_1 C_1 D_1 Z_1\nA_2 B_2 Y_2 C_2 D_2 Z_2\n\\vdots\nA_Q B_Q Y_Q C_Q D_Q Z_Q\n"
            "Output\nPrint Q lines. The i-th line should contain the answer to the i-th query as an integer.\n"
            "Constraints\n- 1\\leq H \\leq 500\n- 1\\leq W \\leq 500\n- 1\\leq F_{i,j} \\leq 10^6\n"
            "- 1\\leq Q\\leq 2\\times 10^5\n- 1\\leq A_i,C_i\\leq H\n- 1\\leq B_i,D_i\\leq W\n"
            "- 1\\leq Y_i\\leq F_{A_i,B_i}\n- 1\\leq Z_i\\leq F_{C_i,D_i}\n- (A_i,B_i,Y_i)\\neq (C_i,D_i,Z_i)\n"
            "- All input values are integers."
        ),
        redundant_bullet="H, W, and the grid of floor counts F are given in the format shown above, followed by Q queries.",
        missing_bullet="The i-th row of the grid corresponds to the i-th block from the north, so a larger row index i is farther south.",
    ),
    # ---------------- LR/MLP binary-probe test-split set ----------------
    dict(
        tag="binary", base_id="amb_task_4", source_problem_id="abc390_d",
        ambiguity_types=["Terminology"],
        prompt=(
            "There are N bags, labeled bag 1, bag 2, \\ldots, bag N.\n"
            "Bag i (1 \\leq i \\leq N) contains A_i stones.\n"
            "Takahashi can perform the following operation any number of times, possibly zero:\n"
            "Choose two bags A and B, and move stones from bag A into bag B.\n"
            "Find the number of different possible values for the following after repeating the operation.\n"
            "- B_1 \\oplus B_2 \\oplus \\cdots \\oplus B_N, where B_i is the final number of stones in bag i.\n"
            "Here, \\oplus denotes bitwise XOR.\n"
            "About bitwise XOR\n"
            "For non-negative integers a and b, the bitwise XOR a \\oplus b is defined as follows:\n"
            "In the binary representation of a \\oplus b, the digit in the 2^k place (k \\ge 0) is 1 if and only if "
            "exactly one of the digits in the 2^k place of a and b is 1; otherwise, it is 0.\n"
            "For example, 3 \\oplus 5 = 6 (in binary, 011 \\oplus 101 = 110).\n"
            "In general, for k non-negative integers x_1, x_2, \\ldots, x_k, their bitwise XOR "
            "x_1 \\oplus x_2 \\oplus \\cdots \\oplus x_k is defined as (\\cdots((x_1 \\oplus x_2) \\oplus x_3) "
            "\\oplus \\cdots) \\oplus x_k, which does not depend on the order of x_1, x_2, \\ldots, x_k.\n"
            "It can be proved that under the constraints of this problem, the number of possible values is finite.\n"
            "Input\nThe input is given from Standard Input in the following format:\nN\nA_1 A_2 \\ldots A_N\n"
            "Output\nPrint the number of different possible values for B_1 \\oplus B_2 \\oplus \\cdots \\oplus B_N "
            "after repeating the operation.\n"
            "Constraints\n- 2 \\leq N \\leq 12\n- 1 \\leq A_i \\leq 10^{17}\n- All input values are integers."
        ),
        redundant_bullet="N is given on the first line, followed by A_1,\\ldots,A_N on the second line, as shown in the input format above.",
        missing_bullet="Moving stones from bag A to bag B moves all of bag A's stones, so bag A becomes empty (0 stones) after the operation.",
    ),
    dict(
        tag="binary", base_id="amb_task_26", source_problem_id="abc391_f",
        ambiguity_types=["Comparison Rules"],
        prompt=(
            "You are given three integer sequences of length N, namely A=(A_1,A_2,\\ldots,A_N), "
            "B=(B_1,B_2,\\ldots,B_N), and C=(C_1,C_2,\\ldots,C_N), and an integer K.\n"
            "For each of the N^3 choices of integers i,j,k (1\\leq i,j,k\\leq N), compute the value "
            "A_iB_j + B_jC_k + C_kA_i. Among all these values, find the K-th value.\n"
            "Input\nThe input is given from Standard Input in the following format:\n"
            "N K\nA_1 A_2 \\ldots A_N\nB_1 B_2 \\ldots B_N\nC_1 C_2 \\ldots C_N\n"
            "Output\nPrint the answer.\n"
            "Constraints\n- 1\\leq N \\leq 2\\times 10^5\n- 1\\leq K \\leq \\min(N^3,5\\times 10^5)\n"
            "- 1\\leq A_i,B_i,C_i \\leq 10^9\n- All input values are integers."
        ),
        redundant_bullet="N, K and the sequences A, B, C are given in the format shown above.",
        missing_bullet="Among the N^3 computed values, find the K-th largest value.",
    ),
    dict(
        tag="binary", base_id="amb_task_49", source_problem_id="abc392_a",
        ambiguity_types=["Collection Semantics"],
        prompt=(
            "You are given a sequence of integers A = (A_1, A_2, A_3).\n"
            "Let B = (B_1, B_2, B_3) be any permutation related of A.\n"
            "Determine whether it is possible that B_1 \\times B_2 = B_3.\n"
            "Input\nThe input is given from Standard Input in the following format:\nA_1 A_2 A_3\n"
            "Output\nIf it is possible that B_1 \\times B_2 = B_3, print Yes; otherwise, print No.\n"
            "Constraints\n- All input values are integers.\n- 1 \\le A_1, A_2, A_3 \\le 100"
        ),
        redundant_bullet="A_1, A_2, A_3 are given on a single line, as shown in the input format above.",
        missing_bullet="B must be a strict permutation of A, containing exactly the same three elements with their order rearranged.",
    ),
    dict(
        tag="binary", base_id="amb_task_87", source_problem_id="abc393_a",
        ambiguity_types=["Behavior"],
        prompt=(
            "There are four types of oysters, labeled 1, 2, 3, and 4. One or some types cause stomach trouble if eaten. "
            "The other types do not cause stomach trouble when eaten.\n"
            "Takahashi ate oysters 1 and 2, and Aoki ate oysters 1 and 3. The information on whether each person got "
            "sick is given as two strings S_1 and S_2. Specifically, S_1 = sick means Takahashi got sick, and "
            "S_1 = fine means Takahashi did not get sick. Likewise, S_2 = sick means Aoki got sick, and S_2 = fine "
            "means Aoki did not get sick.\n"
            "Based on the given information, find which type of oyster causes stomach trouble.\n"
            "Input\nThe input is given from Standard Input in the following format:\nS_1 S_2\n"
            "Output\nPrint the label of the oyster that causes stomach trouble if eaten.\n"
            "Constraints\n- Each of S_1 and S_2 is sick or fine."
        ),
        redundant_bullet="S_1 and S_2 are given on a single line separated by a space, as shown in the input format above.",
        missing_bullet="Exactly one of the four oyster types causes stomach trouble; the other three do not.",
    ),
    dict(
        tag="binary", base_id="amb_task_118", source_problem_id="abc394_c",
        ambiguity_types=["Terminology"],
        prompt=(
            "You are given a string S consisting of uppercase English letters.\n"
            "Apply the following procedure to S, and then output the resulting string:\n"
            "As long as the string contains WA as a substring, repeat the following operation:\n"
            "- Among all occurrences of WA in the string, replace the leftmost one with AC.\n"
            "It can be proved under the constraints of this problem that this operation is repeated at most a finite "
            "number of times.\n"
            "Input\nThe input is given from Standard Input in the following format:\nS\n"
            "Output\nPrint the resulting string after performing the procedure described in the problem statement on S.\n"
            "Constraints\n- S is a string of uppercase English letters with length between 1 and 3\\times 10^5, inclusive."
        ),
        redundant_bullet="S is given as a single line of standard input, as shown in the input format above.",
        missing_bullet="“WA” refers to a contiguous substring, meaning the W and A characters must be adjacent.",
    ),
    dict(
        tag="binary", base_id="amb_task_198", source_problem_id="abc396_c",
        ambiguity_types=["Comparison Rules"],
        prompt=(
            "There are N black balls and M white balls.\n"
            "Each ball has a value. The value of the i-th black ball (1 \\le i \\le N) is B_i, and the value of the "
            "j-th white ball (1 \\le j \\le M) is W_j.\n"
            "Choose zero or more balls so that the number of black balls chosen is the number of white balls chosen. "
            "Among all such choices, find the maximum possible sum of the values of the chosen balls.\n"
            "Input\nThe input is given from Standard Input in the following format:\nN M\nB_1 B_2 \\ldots B_N\nW_1 W_2 \\ldots W_M\n"
            "Output\nPrint the answer.\n"
            "Constraints\n- 1 \\leq N,M \\leq 2\\times 10^5\n- -10^9 \\leq B_i, W_j \\leq 10^9\n- All input values are integers."
        ),
        redundant_bullet="N, M and the value sequences B, W are given in the format shown above.",
        missing_bullet="The number of black balls chosen must be at least the number of white balls chosen, not necessarily equal.",
    ),
    dict(
        tag="binary", base_id="amb_task_212", source_problem_id="abc396_e",
        ambiguity_types=["Indices & Ranges"],
        prompt=(
            "You are given integers N, M and three integer sequences of length M: X = (X_1, X_2, \\ldots, X_M), "
            "Y = (Y_1, Y_2, \\ldots, Y_M), and Z = (Z_1, Z_2, \\ldots, Z_M). \n"
            "We call a length-N sequence of non-negative integers A = (A_1, A_2, \\ldots, A_N) a good sequence if and "
            "only if it satisfies the following condition:\n"
            "- For every integer i with 1 \\le i \\le M, the XOR of A_{X_i} and A_{Y_i} is Z_i.\n"
            "Determine whether a good sequence A=(A_1,A_2,\\ldots,A_N) exists, and if it exists, find one good "
            "sequence that minimizes the sum of its elements \\displaystyle \\sum_{i=1}^N A_i.\n"
            "Notes on XOR\n"
            "For non-negative integers A and B, their XOR A \\oplus B is defined as follows:\n"
            "- In the binary representation of A \\oplus B, the digit in the place corresponding to 2^k \\,(k \\ge 0) "
            "is 1 if and only if exactly one of the digits in the same place of A and B is 1; otherwise, it is 0.\n"
            "For example, 3 \\oplus 5 = 6 (in binary: 011 \\oplus 101 = 110).\n"
            "Input\nThe input is given from Standard Input in the following format:\n"
            "N M\nX_1 Y_1 Z_1\nX_2 Y_2 Z_2\n\\vdots\nX_M Y_M Z_M\n"
            "Output\nIf no good sequence exists, print -1.\n"
            "If a good sequence exists, print one good sequence that minimizes the sum of its elements, separated by spaces.\n"
            "If there are multiple good sequences with the same minimum sum, printing any of them is accepted.\n"
            "Constraints\n- 1 \\le N \\le 2\\times 10^5\n- 0 \\le M \\le 10^5\n- 1 \\le X_i, Y_i \\le N\n"
            "- 0 \\le Z_i \\le 10^9\n- All input values are integers."
        ),
        redundant_bullet="N, M and the sequences X, Y, Z are given in the format shown above.",
        missing_bullet="All X_i and Y_i are guaranteed to satisfy 1 \\leq X_i, Y_i \\leq N.",
    ),
    dict(
        tag="binary", base_id="amb_task_279", source_problem_id="abc398_d",
        ambiguity_types=["Edge Cases"],
        prompt=(
            "There is an infinitely large two-dimensional grid, with a campfire at coordinate (0,0).\n"
            "You are given a length-N string S consisting of N, W, S, E. At times t=1,2,\\dots,N, the following "
            "happen in order:\n"
            "- Wind blows, and all the smoke present at that time moves as follows:\n"
            "- If the t-th character of S is N, smoke in cell (r,c) moves to cell (r-1,c).\n"
            "- If it is W, smoke in cell (r,c) moves to cell (r,c-1).\n"
            "- If it is S, smoke in cell (r,c) moves to cell (r+1,c).\n"
            "- If it is E, smoke in cell (r,c) moves to cell (r,c+1).\n"
            "- If there is no smoke in cell (0,0), new smoke is generated at cell (0,0).\n"
            "Takahashi is standing at cell (R,C).\n"
            "For each integer 1 \\le t \\le N, determine if smoke exists at cell (R,C) at time t+0.5, and print the "
            "response according to the required format.\n"
            "Input\nThe input is given from Standard Input in the following format:\nN R C\nS\n"
            "Output\nPrint an N-character string consisting of 0 and 1.\n"
            "The t-th character (1 \\le t \\le N) should be:\n"
            "- 1 if smoke exists at cell (R,C) at time t+0.5, and\n- 0 otherwise.\n"
            "Constraints\n- N is an integer between 1 and 200000, inclusive.\n"
            "- S is a length N string consisting of N, W, S, E.\n"
            "- R and C are integers between -N and N, inclusive.\n- (R,C) \\neq (0,0)"
        ),
        redundant_bullet="N, R, C and the string S are given in the format shown above.",
        missing_bullet="At time t=0, smoke exists only at cell (0,0); no other cell contains smoke initially.",
    ),
    dict(
        tag="binary", base_id="amb_task_321", source_problem_id="abc400_b",
        ambiguity_types=["Numerical Precision"],
        prompt=(
            "You are given two positive integers N and M.\n"
            "Let X = \\displaystyle\\sum_{i = 0}^{M} N^i. If X, print the value of X. If X, print inf.\n"
            "Input\nThe input is given from Standard Input in the following format:\nN M\n"
            "Output\nPrint the value of X or inf as specified by the problem statement.\n"
            "Constraints\n- 1 \\leq N \\leq 10^9\n- 1 \\leq M \\leq 100\n- All input values are integers."
        ),
        redundant_bullet="N and M are given on a single line, as shown in the input format above.",
        missing_bullet="The threshold value is 10^9; if X > 10^9, print “inf” instead of the exact value.",
    ),
    dict(
        tag="binary", base_id="amb_task_417", source_problem_id="abc400_d",
        ambiguity_types=["Edge Cases"],
        prompt=(
            "Takahashi is about to go buy eel at a fish shop.\n"
            "The town where he lives is divided into a grid of H rows and W columns. Each cell is either a road or a wall.\n"
            "Let us denote the cell at the i-th row from the top (1\\leq i \\leq H) and the j-th column from the left "
            "(1\\leq j \\leq W) as cell (i,j).\n"
            "Information about each cell is given by H strings S_1,S_2,\\ldots,S_H, each of length W. Specifically, "
            "if the j-th character of S_i (1\\leq i \\leq H,1\\leq j\\leq W) is ., cell (i,j) is a road; if it is #, "
            "cell (i,j) is a wall.\n"
            "He can repeatedly perform the following two types of actions in any order:\n"
            "- Move to an adjacent cell (up, down, left, or right) that is within the town and is a road.\n"
            "- Choose one of the four directions (up, down, left, or right) and perform a front kick in that direction.\n"
            "When he performs a front kick, for each of the cells at most 2 steps away in that direction from the "
            "cell he is currently in, if that cell is a wall, it becomes a road.\n"
            "He starts in cell (A,B), and he wants to move to the fish shop in cell (C,D).\n"
            "It is guaranteed that both the cell where he starts and the cell with the fish shop are roads.\n"
            "Find the minimum number of front kicks he needs in order to reach the fish shop.\n"
            "Input\nThe input is given from Standard Input in the following format:\n"
            "H W\nS_1\nS_2\n\\vdots\nS_H\nA B C D\n"
            "Output\nPrint the minimum number of front kicks needed for Takahashi to reach the fish shop.\n"
            "Constraints\n- 1\\leq H\\leq 1000\n- 1\\leq W\\leq 1000\n"
            "- Each S_i is a string of length W consisting of . and #.\n- 1\\leq A,C\\leq H\n- 1\\leq B,D\\leq W\n"
            "- (A,B)\\neq (C,D)\n- H, W, A, B, C, and D are integers.\n"
            "- The cell where Takahashi starts and the cell with the fish shop are roads."
        ),
        redundant_bullet="H, W, the grid strings S_1,\\ldots,S_H, and the coordinates A,B,C,D are given in the format shown above.",
        missing_bullet="A front kick may target cells that extend outside the grid; only the cells inside the grid are checked and changed.",
    ),
]


def make_variant(prompt: str, bullet: str) -> str:
    if "Constraints" in prompt:
        return prompt + "\n- " + bullet
    return prompt + "\nConstraints\n- " + bullet


def main() -> None:
    rows = []
    for item in ITEMS:
        base = item["prompt"]
        common = dict(
            tag=item["tag"],
            base_id=item["base_id"],
            source_problem_id=item["source_problem_id"],
            ambiguity_types=item["ambiguity_types"],
        )
        rows.append(dict(common, id=f"{item['base_id']}__baseline", variant="baseline", label=1, prompt=base))
        rows.append(dict(common, id=f"{item['base_id']}__redundant", variant="redundant", label=1,
                          prompt=make_variant(base, item["redundant_bullet"])))
        rows.append(dict(common, id=f"{item['base_id']}__missing", variant="missing", label=0,
                          prompt=make_variant(base, item["missing_bullet"])))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_mc = sum(1 for r in rows if r["tag"] == "multiclass")
    n_bin = sum(1 for r in rows if r["tag"] == "binary")
    print(f"[build_v2] wrote {len(rows)} rows ({n_mc} multiclass-tag, {n_bin} binary-tag) -> {OUT_PATH}")


if __name__ == "__main__":
    main()
