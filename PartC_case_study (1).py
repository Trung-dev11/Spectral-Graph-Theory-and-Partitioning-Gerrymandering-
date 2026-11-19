"""
=====================================================================
 GROUP 2 - PART C: GERRYMANDERING CASE STUDY (Section 1.5)
=====================================================================
 C1. Build the adjacency matrix A and the Laplacian L of the 16 census
     blocks (2000-2015), compute the Fiedler vector and reproduce the
     partitions of Figure 1.22 (step 1) and Figure 1.23 (step 2).
 C2. Extension: give every block a (hypothetical) population, check
     the population balance of the districts and improve it by moving
     boundary blocks. Report populations BEFORE and AFTER.
 C3. Repeat the partition with the NORMALIZED Laplacian and compare.

 Data: the binary adjacency matrix A printed in the book (p. 33).
 Its sub-matrix on G1 reproduces exactly the matrix L1 of the book
 (p. 36), which the program checks at the start.

 Libraries: numpy (required), matplotlib (only for the figures).
     python -m pip install numpy matplotlib
 Run:
     python PartC_case_study.py
 Figures are saved in the folder  figures/
=====================================================================
"""

import os
import sys
from collections import deque

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Print floats with 3 decimals (report rounding rule)
np.set_printoptions(precision=3, suppress=True, linewidth=120)

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


# =====================================================================
# PART 0. DATA
# =====================================================================
# Each census block is one VERTEX. Row/column k of A <-> block 2000 + k.
BLOCKS = list(range(2000, 2016))

# Binary adjacency matrix from the book (p. 33): A[i, j] = 1 if block i
# and block j share a boundary.
A_BOOK = np.array([
    # 00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15
    [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # 2000
    [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],   # 2001
    [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],   # 2002
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],   # 2003
    [1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0],   # 2004
    [0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0],   # 2005
    [0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],   # 2006
    [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0],   # 2007
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],   # 2008
    [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],   # 2009
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],   # 2010
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1],   # 2011
    [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],   # 2012
    [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1],   # 2013
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],   # 2014
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0],   # 2015
])

# Edge list read off A (upper triangle only, so every edge appears once)
EDGES = [(BLOCKS[i], BLOCKS[j])
         for i in range(len(BLOCKS)) for j in range(i + 1, len(BLOCKS))
         if A_BOOK[i, j] == 1]

# Book results, used only to COMPARE with our results
BOOK_FIEDLER_STEP1 = [0.17, 0.16, 0.2, 0.2, 0.06, -0.1, -0.17, -0.43,
                      -0.64, -0.31, 0.07, 0.14, 0.15, 0.18, 0.15, 0.18]
BOOK_G2 = [2005, 2006, 2007, 2008, 2009]

G1_ORDER_BOOK = [2000, 2001, 2002, 2003, 2004, 2010, 2011, 2012, 2013, 2014, 2015]
L1_BOOK = np.array([
    [ 4, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0],
    [-1,  4, -1,  0, -1,  0,  0, -1,  0,  0,  0],
    [-1, -1,  4, -1,  0,  0,  0,  0, -1,  0,  0],
    [-1,  0, -1,  3,  0,  0,  0,  0, -1,  0,  0],
    [-1, -1,  0,  0,  4,  0, -1, -1,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  2, -1,  0,  0, -1,  0],
    [ 0,  0,  0,  0, -1, -1,  5, -1, -1,  0, -1],
    [ 0, -1,  0,  0, -1,  0, -1,  4, -1,  0,  0],
    [ 0,  0, -1, -1,  0,  0, -1, -1,  5,  0, -1],
    [ 0,  0,  0,  0,  0, -1,  0,  0,  0,  2, -1],
    [ 0,  0,  0,  0,  0,  0, -1,  0, -1, -1,  3],
])
BOOK_EIGENVALUES_L1 = [6.6, 6.4, 5.7, 5.2, 4.2, 4.0, 2.8, 2.3, 2.1, 0.7, 0]
BOOK_FIEDLER_L1 = [-11, -10, -10, -10, -7, 20, 4, -5, -2, 22, 10]   # times 1/sqrt(1499)

# HYPOTHETICAL populations for C2 (the book gives no population data)
POPULATION = {
    2000: 310, 2001: 180, 2002: 150, 2003: 220, 2004: 140,
    2005: 120, 2006: 110, 2007: 130, 2008: 160, 2009: 125,
    2010: 170, 2011: 190, 2012: 135, 2013: 260, 2014: 150,
    2015: 175,
}

# Approximate positions of the blocks on the map of Figure 1.23 (drawing only)
POSITION = {
    2000: (4.2, 6.6), 2001: (4.3, 5.9), 2002: (4.6, 6.3), 2003: (5.1, 6.6),
    2004: (4.0, 5.3), 2005: (3.6, 4.4), 2006: (3.4, 5.0), 2007: (2.9, 4.4),
    2008: (2.3, 3.8), 2009: (3.2, 3.9), 2010: (3.7, 3.7), 2011: (4.5, 4.4),
    2012: (4.4, 5.3), 2013: (5.2, 4.8), 2014: (4.5, 3.0), 2015: (4.9, 3.7),
}


# =====================================================================
# PART 1. HELPER FUNCTIONS
# =====================================================================
def adjacency_matrix(nodes, edges):
    """A[i, j] = 1 if nodes[i] and nodes[j] are adjacent.

    Only edges with BOTH ends in 'nodes' are kept, so the same function
    also gives the adjacency matrix of a subgraph (for example G1).
    """
    pos = {v: i for i, v in enumerate(nodes)}
    A = np.zeros((len(nodes), len(nodes)))
    for u, v in edges:
        if u in pos and v in pos:
            A[pos[u], pos[v]] = 1
            A[pos[v], pos[u]] = 1          # undirected graph -> symmetric A
    return A


def laplacian(A):
    """L = D - A, where D is the diagonal matrix of vertex degrees."""
    return np.diag(A.sum(axis=1)) - A


def normalized_laplacian(A):
    """Normalized Laplacian T L T with T = D^(-1/2)."""
    degrees = A.sum(axis=1)
    if np.any(degrees == 0):
        raise ValueError("Isolated vertex (degree 0): D^(-1/2) is undefined.")
    T = np.diag(1.0 / np.sqrt(degrees))
    return T @ laplacian(A) @ T


def fiedler_vector(M):
    """Return (eigenvalues in increasing order, Fiedler vector).

    np.linalg.eigh is for SYMMETRIC matrices: real eigenvalues sorted in
    increasing order, orthonormal eigenvectors stored as COLUMNS.
    v and -v are both eigenvectors; we fix the sign so that the last
    non-zero entry is positive (as in the book). The sign does not
    change the partition.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    v2 = eigenvectors[:, 1]
    last = v2[np.abs(v2) > 1e-12][-1]
    if last < 0:
        v2 = -v2
    return eigenvalues, v2


def warn_near_zero(nodes, v, threshold=0.01):
    """Print the vertices whose Fiedler entry is almost 0. Such a vertex lies
    on the border between the two parts: a tiny change in the data could
    move it to the other side, so its assignment is fragile."""
    close = [(node, x) for node, x in zip(nodes, v) if abs(x) < threshold]
    for node, x in close:
        print(f"  WARNING: block {node} has entry {x:+.3f} (|entry| < {threshold})"
              " -> fragile, it sits on the border")


def smaller_first(part_a, part_b):
    """Return (smaller part, larger part). Naming the parts by SIZE (not by
    the sign of the vector) keeps the names correct even if the sign flips."""
    return (part_a, part_b) if len(part_a) <= len(part_b) else (part_b, part_a)


def split_by_sign(nodes, v):
    """Theorem 2: negative entries -> one part, non-negative -> the other."""
    negative = [node for node, x in zip(nodes, v) if x < 0]
    non_negative = [node for node, x in zip(nodes, v) if x >= 0]
    return negative, non_negative


def fiedler_split(nodes, edges, normalized=False):
    """A -> L (or normalized L) -> Fiedler vector -> split by sign."""
    A = adjacency_matrix(nodes, edges)
    M = normalized_laplacian(A) if normalized else laplacian(A)
    eigenvalues, v2 = fiedler_vector(M)
    part_neg, part_pos = split_by_sign(nodes, v2)
    return eigenvalues, v2, part_neg, part_pos


def cut_edges(part_a, part_b, edges):
    """Edges with one end in part_a and the other end in part_b."""
    a, b = set(part_a), set(part_b)
    return [(u, v) for u, v in edges if (u in a and v in b) or (u in b and v in a)]


def is_connected(nodes, edges):
    """Contiguity check by breadth-first search (BFS)."""
    nodes = set(nodes)
    if not nodes:
        return False
    neighbors = {v: [] for v in nodes}
    for u, v in edges:
        if u in nodes and v in nodes:
            neighbors[u].append(v)
            neighbors[v].append(u)
    start = next(iter(nodes))
    seen, queue = {start}, deque([start])
    while queue:
        for w in neighbors[queue.popleft()]:
            if w not in seen:
                seen.add(w)
                queue.append(w)
    return seen == nodes


def same_partition(p, q):
    """True if two partitions contain the same parts (order ignored)."""
    return sorted(map(sorted, p)) == sorted(map(sorted, q))


def population_table(districts, title):
    """Print population of each district and deviation from the ideal."""
    total = sum(POPULATION.values())
    ideal = total / len(districts)
    print(f"\n{title}")
    print(f"  Total population = {total}, ideal per district = {ideal:.1f}")
    print(f"  {'District':<10}{'Population':>11}{'Deviation (%)':>15}   Blocks")
    for i, d in enumerate(districts, 1):
        pop = sum(POPULATION[b] for b in d)
        print(f"  {i:<10}{pop:>11}{100 * (pop - ideal) / ideal:>+15.1f}   {sorted(d)}")
    pops = [sum(POPULATION[b] for b in d) for d in districts]
    print(f"  Largest - smallest = {max(pops) - min(pops)}")


def imbalance(districts):
    """Sum of squared deviations from the ideal population."""
    ideal = sum(POPULATION.values()) / len(districts)
    return sum((sum(POPULATION[b] for b in d) - ideal) ** 2 for d in districts)


def balance_by_moving_boundary_blocks(districts, edges, max_moves=10):
    """C2: greedily move ONE boundary block at a time.

    In every round all valid moves are tried and the one that reduces
    the imbalance most is applied. A move is valid if
      - the block is adjacent to the receiving district, and
      - the giving district stays non-empty and contiguous.
    Stop when no move improves the balance.
    """
    edge_set = set(edges) | {(v, u) for u, v in edges}
    districts = [set(d) for d in districts]
    for step in range(1, max_moves + 1):
        best = None
        current = imbalance(districts)
        for i, source in enumerate(districts):
            for block in sorted(source):
                remaining = source - {block}
                if not remaining or not is_connected(remaining, edges):
                    continue
                for j, target in enumerate(districts):
                    if j == i or not any((block, t) in edge_set for t in target):
                        continue
                    trial = [set(d) for d in districts]
                    trial[i].remove(block)
                    trial[j].add(block)
                    score = imbalance(trial)
                    if score < current and (best is None or score < best[0]):
                        best = (score, block, i, j)
        if best is None:
            break
        _, block, i, j = best
        districts[i].remove(block)
        districts[j].add(block)
        print(f"  Move {step}: block {block} from district {i + 1} to district {j + 1}")
    return [sorted(d) for d in districts]


def draw_districts(districts, edges, filename, title):
    """Draw the graph on the plane, one grey level per district."""
    try:
        import matplotlib
        matplotlib.use("Agg")                 # save to file, no window
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"  (matplotlib not installed -> figure {filename} skipped)")
        return
    os.makedirs(FIG_DIR, exist_ok=True)
    colors = ["#9e9e9e", "#d9d9d9", "#5f5f5f", "#bcbcbc"]
    fig, ax = plt.subplots(figsize=(6, 5))
    for u, v in edges:
        (x1, y1), (x2, y2) = POSITION[u], POSITION[v]
        ax.plot([x1, x2], [y1, y2], color="black", linewidth=0.8, zorder=1)
    for k, d in enumerate(districts):
        c = colors[k % len(colors)]
        for b in d:
            x, y = POSITION[b]
            ax.scatter(x, y, s=650, color=c, edgecolors="black", zorder=2)
            ax.text(x, y, str(b), ha="center", va="center", fontsize=7, zorder=3,
                    color="white" if c == "#5f5f5f" else "black")
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, filename)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  Figure saved: {path}")


# =====================================================================
# PART 2. MAIN PROGRAM
# =====================================================================
if __name__ == "__main__":

    # ---------------------------------------------------------------
    # DATA CHECK
    # ---------------------------------------------------------------
    print("=" * 66)
    print("DATA CHECK")
    print("=" * 66)
    print("A symmetric:", np.array_equal(A_BOOK, A_BOOK.T),
          "| number of edges:", len(EDGES))
    L1_from_A = laplacian(adjacency_matrix(G1_ORDER_BOOK, EDGES))
    print("Sub-matrix of A on G1 gives exactly L1 of the book:",
          np.array_equal(L1_from_A, L1_BOOK))

    # ---------------------------------------------------------------
    # C1 - STEP 1: split the 16 blocks into 2 parts (Figure 1.22)
    # ---------------------------------------------------------------
    print("\n" + "=" * 66)
    print("C1 - STEP 1: SPLIT 16 BLOCKS INTO 2 PARTS (Figure 1.22)")
    print("=" * 66)
    L = laplacian(A_BOOK.astype(float))
    print("Degrees (diagonal of L):", np.diag(L).astype(int))

    eigenvalues, v2, part_neg, part_pos = fiedler_split(BLOCKS, EDGES)
    print("\nEigenvalues of L:", eigenvalues)
    print(f"lambda_1 = {abs(eigenvalues[0]):.3f},"
          f" lambda_2 = a(G) = {eigenvalues[1]:.3f} (> 0, so the graph is connected)")
    print(f"{'Block':<7}{'Fiedler (ours)':>15}{'Book':>8}")
    for block, value, book in zip(BLOCKS, v2, BOOK_FIEDLER_STEP1):
        print(f"{block:<7}{value:>+15.3f}{book:>+8.2f}")
    print("Max difference with the book:",
          f"{np.max(np.abs(v2 - np.array(BOOK_FIEDLER_STEP1))):.3f}")

    warn_near_zero(BLOCKS, v2)
    G2, G1 = smaller_first(part_neg, part_pos)   # G2 = smaller part (as in the book)
    print("G1 =", G1)
    print("G2 =", G2)
    print("Same as the book (G2 = 2005..2009):", G2 == BOOK_G2)
    print("Cut edges:", cut_edges(G1, G2, EDGES))
    print("G1 contiguous:", is_connected(G1, EDGES),
          "| G2 contiguous:", is_connected(G2, EDGES))

    # ---------------------------------------------------------------
    # C1 - STEP 2: split G1 again (Figure 1.23)
    # ---------------------------------------------------------------
    print("\n" + "=" * 66)
    print("C1 - STEP 2: SPLIT G1 AGAIN (Figure 1.23)")
    print("=" * 66)
    # Split the G1 we just COMPUTED (not the list typed from the book), so a
    # different step 1 would automatically give a different step 2.
    # G1 is in increasing order, which is also the order of L1 in the book.
    print("G1 computed above has the same order as L1 in the book:",
          G1 == G1_ORDER_BOOK)
    ev1, v_g1, part1, part2 = fiedler_split(G1, EDGES)
    warn_near_zero(G1, v_g1)
    print("Eigenvalues of L1 (decreasing):", np.round(ev1[::-1], 1))
    print("Book                          :", BOOK_EIGENVALUES_L1)
    print("sqrt(1499) * v2 (ours) =", np.round(np.sqrt(1499) * v_g1, 1))
    print("Book                   =", BOOK_FIEDLER_L1)
    print("District A:", part1)
    print("District B:", part2)
    print("Cut edges:", cut_edges(part1, part2, EDGES))

    three_districts = [part1, part2, G2]
    for i, d in enumerate(three_districts, 1):
        print(f"District {i} contiguous: {is_connected(d, EDGES)}")
    draw_districts(three_districts, EDGES, "C1_three_districts.png",
                   "C1: three districts from Fiedler vectors")

    # ---------------------------------------------------------------
    # C2: POPULATION BALANCE
    # ---------------------------------------------------------------
    print("\n" + "=" * 66)
    print("C2 - POPULATION BALANCE (HYPOTHETICAL populations)")
    print("=" * 66)
    population_table(three_districts, "BEFORE balancing:")
    print("\nBalancing by moving boundary blocks:")
    balanced = balance_by_moving_boundary_blocks(three_districts, EDGES)
    population_table(balanced, "AFTER balancing:")
    for i, d in enumerate(balanced, 1):
        print(f"  District {i} still contiguous: {is_connected(d, EDGES)}")
    draw_districts(balanced, EDGES, "C2_balanced_districts.png",
                   "C2: districts after population balancing")

    # ---------------------------------------------------------------
    # C3: NORMALIZED LAPLACIAN
    # ---------------------------------------------------------------
    print("\n" + "=" * 66)
    print("C3 - COMPARISON WITH THE NORMALIZED LAPLACIAN")
    print("=" * 66)
    ev_n, vn, n_neg, n_pos = fiedler_split(BLOCKS, EDGES, normalized=True)
    print("Eigenvalues of the normalized Laplacian (always in [0, 2]):", ev_n)
    print(f"lambda_2 (normalized) = {ev_n[1]:.3f}")
    print(f"{'Block':<7}{'Fiedler (normalized)':>21}")
    for block, value in zip(BLOCKS, vn):
        print(f"{block:<7}{value:>+21.3f}")
    warn_near_zero(BLOCKS, vn)
    G2n, G1n = smaller_first(n_neg, n_pos)
    print("Step 1 - normalized:  ", G1n, "|", G2n)
    print("Step 1 - unnormalized:", G1, "|", G2)
    print("Step 1 identical:", same_partition([G1n, G2n], [G1, G2]))

    _, vn1, m1, m2 = fiedler_split(G1n, EDGES, normalized=True)
    warn_near_zero(G1n, vn1)
    print("\nStep 2 - normalized:  ", m1, "|", m2)
    print("Step 2 - unnormalized:", part1, "|", part2)
    same = same_partition([m1, m2], [part1, part2])
    print("Step 2 identical:", same)
    if not same:
        ref = part1 if len(set(m1) & set(part1)) >= len(set(m1) & set(part2)) else part2
        print("Blocks that change district:", sorted(set(m1) ^ set(ref)))
    normalized_districts = [m1, m2, G2n]
    for i, d in enumerate(normalized_districts, 1):
        print(f"District {i} contiguous: {is_connected(d, EDGES)}")
    population_table(normalized_districts, "Populations of the normalized partition:")
    draw_districts(normalized_districts, EDGES, "C3_normalized_districts.png",
                   "C3: districts from the normalized Laplacian")
