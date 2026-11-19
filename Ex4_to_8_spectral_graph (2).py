"""
===========================================================================
 Group 2 - Spectral Graph Theory and Partitioning (Gerrymandering)
 Part B: Exercises 4, 5, 6, 7 and 8 (Section 1.6)
 Book: C. Arangala, "Linear Algebra With Machine Learning and Data",
       CRC Press, 2023.
===========================================================================

HOW TO RUN
    1. Install the three libraries (only once). In the VS Code terminal:
           python -m pip install numpy networkx matplotlib
    2. Run this file:
           python Ex4_to_8_spectral_graph.py
    3. Results are printed in the terminal. Figures are saved as PNG files
       in the folder "figures" (created next to this file).

LIBRARIES
    Only numpy, networkx and matplotlib are used (plus "itertools" and
    "os", which are part of Python itself, so nothing to install).
    - networkx  : stores each graph (vertices + edges) and draws it
    - numpy     : matrices, eigenvalues and eigenvectors (numpy.linalg.eigh)
    - matplotlib: plots

DATA SOURCE
    The edge lists below were read from Figures 1.24, 1.25 and 1.26 and
    cross-checked against the adjacency / Laplacian matrices printed in the
    book's "Sample Solutions to Exercises" (pages 265-268).

OVERALL IDEA (the same recipe is used in every exercise)
    graph  ->  adjacency matrix A  ->  Laplacian L = D - A
           ->  eigenvalues/eigenvectors of L
           ->  2nd smallest eigenvalue  = Fiedler value  lambda_2
               its eigenvector          = Fiedler vector v_2
           ->  split vertices by the SIGN of v_2  (negative | positive)
===========================================================================
"""

import os
import sys
from itertools import combinations

import numpy as np
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# GENERAL SETTINGS
# ---------------------------------------------------------------------------
# Print Unicode symbols (lambda, theta, ...) correctly on Windows terminals.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Show numbers with 3 decimals, no scientific notation, wide lines.
np.set_printoptions(precision=3, suppress=True, linewidth=200)

# Set to True if you also want the figure windows to pop up.
SHOW_FIGURES = False
if not SHOW_FIGURES:
    matplotlib.use("Agg")          # draw "in memory", only save PNG files

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

TOL = 1e-9                         # numbers smaller than this count as 0


# ===========================================================================
# 1. DATA
# ===========================================================================

# Exercise 4 & 5 - Figure 1.24 (10 vertices, 11 edges)
#   - a cycle 1-2-3-4-5-6-7-1
#   - a triangle 8-9-10
#   - a single "bridge" edge 1-8 joining the two parts
EDGES_EX4 = [
    (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 1),   # cycle
    (1, 8),                                                   # bridge
    (8, 9), (9, 10), (10, 8),                                 # triangle
]

# Exercise 6 - Figure 1.25: unweighted marmot network (15 marmots, 29 edges)
EDGES_EX6 = [
    (1, 2), (1, 3), (1, 5), (1, 6),
    (2, 3), (2, 5),
    (3, 5), (3, 6), (3, 7),
    (4, 6),
    (5, 7), (5, 8),
    (6, 8),
    (7, 9), (7, 10), (7, 11), (7, 13), (7, 14),
    (9, 11), (9, 13),
    (10, 13), (10, 14),
    (11, 13), (11, 15),
    (12, 13), (12, 14), (12, 15),
    (13, 14),
    (14, 15),
]

# Exercise 8 - Figure 1.26: SAME 29 edges, now with a weight
#   weight = number of interactions between the two marmots in one week
#   Weights READ FROM FIGURE 1.26 (book p. 40). With these weights we get
#   exactly the book's answer 8b (lambda_2 = 2.38721, v_7 = -0.02).
#   NOTE: the Laplacian printed in the book's Sample Solution 8a disagrees
#   with Figure 1.26 on 4 edges (erratum). See BOOK_8A_WEIGHTS below.
EDGES_EX8 = [
    (1, 2, 10), (1, 3, 3), (1, 5, 6), (1, 6, 10),
    (2, 3, 6), (2, 5, 4),
    (3, 5, 3), (3, 6, 15), (3, 7, 13),
    (4, 6, 13),
    (5, 7, 12), (5, 8, 7),
    (6, 8, 13),
    (7, 9, 1), (7, 10, 1), (7, 11, 1), (7, 13, 9), (7, 14, 5),
    (9, 11, 11), (9, 13, 8),
    (10, 13, 6), (10, 14, 3),
    (11, 13, 11), (11, 15, 10),
    (12, 13, 15), (12, 14, 12), (12, 15, 15),
    (13, 14, 10),
    (14, 15, 14),
]

# The 4 edges where the Laplacian of Sample Solution 8a differs from
# Figure 1.26:  edge -> weight implied by the book's 8a matrix
BOOK_8A_WEIGHTS = {(5, 7): 1, (7, 9): 5, (7, 10): 5, (7, 11): 5}

# Book's answer 8b, for comparison only
BOOK_8B_LAMBDA2 = 2.38721

# Exercise 7 - a SMALL HYPOTHETICAL example (made up by us, not from the
# book) used only to illustrate a DIRECTED dominance graph.
#   (i, j, w) means: marmot i showed dominance over marmot j, w times.
EDGES_EX7_DOMINANCE = [
    (1, 2, 4), (1, 3, 2), (2, 3, 5), (3, 1, 1),
    (4, 5, 3), (5, 6, 2), (6, 4, 2), (3, 4, 1),
]


# ===========================================================================
# 2. BASIC TOOLS
# ===========================================================================

def build_graph(edges, weighted=False):
    """Create an undirected networkx Graph from a list of edges.

    edges = [(u, v), ...]        if weighted is False (every weight = 1)
    edges = [(u, v, w), ...]     if weighted is True
    """
    G = nx.Graph()
    for e in edges:
        w = e[2] if weighted else 1
        G.add_edge(e[0], e[1], weight=w)
    return G


def adjacency_matrix(G):
    """Adjacency matrix A: A[i, j] = weight of edge (i, j), 0 if no edge.

    Vertices are put in increasing order 1, 2, ..., n so that
    row/column k-1 of the matrix belongs to vertex k.
    """
    nodes = sorted(G.nodes())
    A = nx.to_numpy_array(G, nodelist=nodes, weight="weight")
    return nodes, A


def laplacian_matrix(A):
    """Laplacian L = D - A.

    D is the diagonal "degree matrix": D[i, i] = sum of row i of A
    (= number of neighbours, or total weight if the graph is weighted).
    """
    D = np.diag(A.sum(axis=1))
    return D - A


def fiedler(A):
    """Return (all eigenvalues of L, Fiedler value lambda_2, Fiedler vector v_2).

    numpy.linalg.eigh is made for SYMMETRIC matrices (L is symmetric):
      - eigenvalues come out sorted from smallest to largest
      - eigenvectors are the COLUMNS of 'vecs' and already have length 1
    So column 0 belongs to lambda_1 = 0, column 1 belongs to lambda_2.
    """
    L = laplacian_matrix(A)
    vals, vecs = np.linalg.eigh(L)
    v2 = vecs[:, 1]

    # An eigenvector is only defined up to its sign (v and -v are both
    # correct). We fix the sign so that the last non-zero entry is positive.
    # This matches the book's answers and makes every run print the same
    # numbers. The CLUSTERS do not depend on this choice.
    last = np.flatnonzero(np.abs(v2) > TOL)[-1]
    if v2[last] < 0:
        v2 = -v2

    # Safety checks (they print nothing when everything is fine):
    #  - lambda_2 = 0 means the graph is NOT connected (part A3), so the
    #    sign split would only separate the connected components.
    #  - lambda_2 = lambda_3 means the Fiedler vector is not unique, so the
    #    split could change from one computer to another.
    if vals[1] <= TOL:
        print("    WARNING: lambda_2 = 0 -> the graph is not connected.")
    if len(vals) > 2 and vals[2] - vals[1] < 1e-6:
        print("    WARNING: lambda_2 = lambda_3 -> Fiedler vector not unique.")
    return vals, vals[1], v2


def fiedler_split(nodes, A, v2=None):
    """Split the vertices into two groups by the sign of the Fiedler vector.
    If v2 was already computed it can be passed in (avoids computing twice)."""
    if v2 is None:
        _, _, v2 = fiedler(A)
    negative = [v for v, x in zip(nodes, v2) if x < 0]
    positive = [v for v, x in zip(nodes, v2) if x >= 0]
    return negative, positive


def cut_size(nodes, A, V1):
    """|boundary(V1)| = total weight of the edges with ONE end in V1
    and the other end outside V1 (the edges we 'cut')."""
    inside = np.array([v in V1 for v in nodes])
    return A[np.ix_(inside, ~inside)].sum()


def isoperimetric_ratio(nodes, A, V1):
    """Return the two versions of the isoperimetric ratio used in the book.

      Definition 1 (book text, p. 25):     |boundary(V1)| / |V1|
      Definition 2 (formula for theta_G and
                    the Sample Solutions): |boundary(V1)| / (|V1| * |V2|)

    Our group CHOSE Definition 2 in part A6 (it is the one linked to
    x^T L x and to the Fiedler value). Definition 1 is printed only to
    show that both lead to the same conclusion here.
    """
    cut = cut_size(nodes, A, V1)
    n1 = len(V1)
    n2 = len(nodes) - n1
    return cut, cut / n1, cut / (n1 * n2)


def isoperimetric_number(nodes, A):
    """theta_G = smallest isoperimetric ratio over ALL sets V1 with
    1 <= |V1| <= n/2  (brute force: we really try every subset).

    Returns, for both versions, (minimum value, list of best V1 sets).
    Only possible for small graphs (2^n subsets grows very fast).
    """
    n = len(nodes)
    best = {1: (np.inf, []), 2: (np.inf, [])}
    for size in range(1, n // 2 + 1):
        for V1 in combinations(nodes, size):
            _, r1, r2 = isoperimetric_ratio(nodes, A, V1)
            for version, r in ((1, r1), (2, r2)):
                value, sets = best[version]
                if r < value - TOL:                 # new minimum
                    best[version] = (r, [list(V1)])
                elif abs(r - value) <= TOL:         # tie
                    sets.append(list(V1))
    return best


def subgraph_matrix(nodes, A, keep):
    """Adjacency matrix of the subgraph that only keeps the vertices 'keep'."""
    idx = [nodes.index(v) for v in keep]
    return sorted(keep), A[np.ix_(idx, idx)]


# ===========================================================================
# 3. PRINTING AND DRAWING HELPERS
# ===========================================================================

def title(text):
    print("\n" + "=" * 75)
    print(text)
    print("=" * 75)


def print_matrix(name, M):
    """Print a matrix with row/column labels 1..n (integers if possible)."""
    n = M.shape[0]
    is_int = np.allclose(M, np.round(M))
    width = 4
    print(f"{name} =")
    print("     " + "".join(f"{j:>{width}}" for j in range(1, n + 1)))
    for i in range(n):
        if is_int:
            row = "".join(f"{int(round(x)):>{width}}" for x in M[i])
        else:
            row = "".join(f"{x:>{width}.1f}" for x in M[i])
        print(f"{i + 1:>3}: " + row)


def draw_clusters(ax, G, pos, clusters, show_weights=False, heading=""):
    """Draw graph G on axis 'ax', one colour per cluster.
    Edges that are cut (between two clusters) are drawn red and dashed."""
    colours = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
    colour_of = {}
    for k, c in enumerate(clusters):
        for v in c:
            colour_of[v] = colours[k % len(colours)]
    group_of = {v: k for k, c in enumerate(clusters) for v in c}

    kept = [e for e in G.edges() if group_of[e[0]] == group_of[e[1]]]
    cut = [e for e in G.edges() if group_of[e[0]] != group_of[e[1]]]
    nx.draw_networkx_edges(G, pos, edgelist=kept, ax=ax, edge_color="grey")
    nx.draw_networkx_edges(G, pos, edgelist=cut, ax=ax, edge_color="red",
                           style="dashed", width=1.8)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=420,
                           node_color=[colour_of[v] for v in G.nodes()])
    nx.draw_networkx_labels(G, pos, ax=ax, font_color="white",
                            font_size=9, font_weight="bold")
    if show_weights:
        labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, ax=ax,
                                     font_size=7)
    ax.set_title(heading, fontsize=10)
    ax.axis("off")


def draw_fiedler_bars(ax, nodes, v2, heading=""):
    """Bar chart of the Fiedler vector: blue = negative, orange = positive."""
    colours = ["#4C72B0" if x < 0 else "#DD8452" for x in v2]
    ax.bar([str(v) for v in nodes], v2, color=colours)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("vertex")
    ax.set_ylabel("entry of $v_2$")
    ax.set_title(heading, fontsize=10)


def save(fig, filename):
    path = os.path.join(FIG_DIR, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)
    print(f"[figure saved: figures/{filename}]")


# Positions of the vertices, chosen to look like the figures in the book.
POS_EX4 = {1: (-1, 0), 2: (0, 0.8), 3: (1.2, 1.0), 4: (2.2, 0.6),
           5: (2.2, -0.4), 6: (1.2, -0.9), 7: (0, -0.7),
           8: (-2.3, 0), 9: (-3.2, 0.6), 10: (-3.2, -0.5)}


def circle_positions(n=15):
    """Marmots on a circle: v1 top-left, going counter-clockwise (Fig 1.25)."""
    pos = {}
    for k in range(1, n + 1):
        angle = np.deg2rad(102 + (k - 1) * 24)
        pos[k] = (np.cos(angle), np.sin(angle))
    return pos


POS_MARMOT = circle_positions()


# ===========================================================================
# 4. EXERCISES
# ===========================================================================

def exercise_4():
    title("EXERCISE 4 - Figure 1.24: adjacency, Laplacian, Fiedler value/vector")
    G = build_graph(EDGES_EX4)
    nodes, A = adjacency_matrix(G)

    # (a) adjacency matrix
    print("(a)")
    print_matrix("A", A)

    # (b) Laplacian matrix
    L = laplacian_matrix(A)
    print("\n(b)")
    print_matrix("L = D - A", L)
    # Quick check: every row of L sums to 0  (so L * [1,1,...,1] = 0)
    print("Check: row sums of L =", L.sum(axis=1))

    # (c) Fiedler value and Fiedler vector
    vals, lam2, v2 = fiedler(A)
    print("\n(c) Eigenvalues of L (smallest -> largest):")
    print("   ", vals)
    # abs(): the computer may return -0.000 (tiny round-off error)
    print(f"    lambda_1 = {abs(vals[0]):.3f}  (always 0)")
    print(f"    Fiedler value  lambda_2 = {lam2:.3f}  (> 0, so G is connected)")
    print("    Fiedler vector v_2 (unit length):")
    for v, x in zip(nodes, v2):
        print(f"      vertex {v:>2}: {x:+.3f}")

    # (d) clustering by sign
    neg, pos = fiedler_split(nodes, A)
    print("\n(d) Split by the sign of v_2:")
    print("    V1 (v_2 < 0)  =", neg)
    print("    V2 (v_2 >= 0) =", pos)
    print("    number of cut edges =", int(cut_size(nodes, A, neg)))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    draw_clusters(axes[0], G, POS_EX4, [neg, pos],
                  heading="Ex 4: two clusters from the Fiedler vector")
    draw_fiedler_bars(axes[1], nodes, v2,
                      heading=f"Fiedler vector ($\\lambda_2$ = {lam2:.3f})")
    save(fig, "ex4_fiedler.png")
    return nodes, A, neg, pos


def exercise_5(nodes, A, fiedler_neg, fiedler_pos):
    title("EXERCISE 5 - isoperimetric number theta_G of Figure 1.24")
    # Smaller half of the Fiedler split plays the role of V1 (|V1| <= n/2).
    V_small = min(fiedler_neg, fiedler_pos, key=len)
    cut, r1, r2 = isoperimetric_ratio(nodes, A, V_small)
    print("Fiedler partition: V1 =", V_small)
    print(f"   cut = {int(cut)},  cut/|V1| = {r1:.3f},  "
          f"cut/(|V1||V2|) = {int(cut)}/{len(V_small) * (len(nodes) - len(V_small))}"
          f" = {r2:.3f}")

    best = isoperimetric_number(nodes, A)
    for version, text in ((1, "Def 1: cut/|V1|"),
                          (2, "Def 2: cut/(|V1||V2|)  [our choice]")):
        value, sets = best[version]
        print(f"theta_G with {text:<36}= {value:.3f}   at V1 = {sets}")

    best_set = best[2][1][0]
    same = sorted(best_set) == sorted(V_small)
    print("\nConclusion:")
    print("   Minimising theta_G cuts the graph into", best_set, "and",
          [v for v in nodes if v not in best_set], "(only 1 edge, 1-8, is cut).")
    print("   The Fiedler vector puts vertex 1 together with 8, 9, 10 (2 edges cut).")
    print("   => The two partitions", "AGREE." if same else "DO NOT agree.")
    print("   theta_G (Definition 2) = 1/(3*7) = 1/21, the same as the book.")
    print("   Both definitions give the same best set here.")

    # Figure: the two partitions side by side
    G = build_graph(EDGES_EX4)
    other = [v for v in nodes if v not in best_set]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    draw_clusters(axes[0], G, POS_EX4, [fiedler_neg, fiedler_pos],
                  heading=f"Fiedler split: cut = 2, ratio = {r2:.3f}")
    draw_clusters(axes[1], G, POS_EX4, [other, best_set],
                  heading=f"Minimum theta_G: cut = 1, ratio = {best[2][0]:.3f}")
    save(fig, "ex5_fiedler_vs_theta.png")


def k_step_reachability(nodes, A, k):
    """Return S_k = I + A + A^2 + ... + A^k  and the 'hubs':
    vertices whose row of S_k has NO zero entry.
    (A^m)[i, j] = number of walks of length m from i to j, so
    S_k[i, j] > 0  <=>  j can be reached from i in at most k steps."""
    n = len(nodes)
    S = np.eye(n)
    power = np.eye(n)
    for _ in range(k):
        power = power @ A
        S = S + power
    hubs = [nodes[i] for i in range(n) if np.all(S[i] > 0)]
    return S, hubs


def recursive_fiedler_4(nodes, A):
    """Ex 6(g): split G into 2 with the Fiedler vector of G, then split EACH
    half again with the Fiedler vector of its own subgraph -> 4 clusters."""
    first_neg, first_pos = fiedler_split(nodes, A)
    clusters = []
    details = []
    for half in (first_neg, first_pos):
        sub_nodes, sub_A = subgraph_matrix(nodes, A, half)
        vals, lam2, v2 = fiedler(sub_A)
        if lam2 <= TOL:          # subgraph not connected -> split is meaningless
            print(f"    WARNING: subgraph {sub_nodes} is not connected.")
        neg, pos = fiedler_split(sub_nodes, sub_A, v2)   # reuse v2, no recompute
        clusters += [neg, pos]
        details.append((sub_nodes, lam2, v2, neg, pos))
    return clusters, details


def total_cut(nodes, A, clusters):
    """Total weight of edges whose two ends lie in different clusters."""
    group = {v: k for k, c in enumerate(clusters) for v in c}
    i_of = {v: i for i, v in enumerate(nodes)}
    total = 0
    for u in nodes:
        for v in nodes:
            if u < v and group[u] != group[v]:
                total += A[i_of[u], i_of[v]]
    return total


def exercise_6():
    title("EXERCISE 6 - Marmot social network (Figure 1.25, unweighted)")
    G = build_graph(EDGES_EX6)
    nodes, A = adjacency_matrix(G)
    print(f"{G.number_of_nodes()} marmots, {G.number_of_edges()} edges, "
          f"connected: {nx.is_connected(G)}")

    # (a)
    print("\n(a)")
    print_matrix("A", A)

    # (b) powers of A
    print("\n(b) Reachability with S_k = I + A + ... + A^k")
    for k in (1, 2, 3):
        S, hubs = k_step_reachability(nodes, A, k)
        print(f"    within {k} step(s): marmots linked to ALL others = "
              f"{hubs if hubs else 'none'}")
    S2, _ = k_step_reachability(nodes, A, 2)
    i7 = nodes.index(7)
    missing = [nodes[j] for j in range(len(nodes)) if S2[i7, j] == 0]
    print(f"    (within 2 steps, marmot 7 misses only marmot(s) {missing})")

    # (c) Laplacian
    L = laplacian_matrix(A)
    print("\n(c)")
    print_matrix("L = D - A", L)

    # (d) isoperimetric ratio of the given set
    V = [1, 2, 3, 5, 6, 7]
    cut, r1, r2 = isoperimetric_ratio(nodes, A, V)
    n2 = len(nodes) - len(V)
    print(f"\n(d) V = {V}")
    print(f"    cut edges |boundary(V)| = {int(cut)}")
    print(f"    Definition 2: cut/(|V||V2|) = {int(cut)}/({len(V)}*{n2}) = {r2:.3f}"
          "  (= 4/27, the book's answer)")
    print(f"    Definition 1: cut/|V|       = {int(cut)}/{len(V)} = {r1:.3f}")

    # (e) Fiedler vector
    vals, lam2, v2 = fiedler(A)
    print(f"\n(e) Fiedler value lambda_2 = {lam2:.3f}")
    print("    Fiedler vector v_2 =", v2)

    # (f) two clusters
    neg, pos = fiedler_split(nodes, A)
    print("\n(f) Two clusters by sign of v_2:")
    print("    V1 (v_2 < 0)  =", neg)
    print("    V2 (v_2 >= 0) =", pos)
    cut_f, _, r2_f = isoperimetric_ratio(nodes, A, neg)
    print(f"    cut edges = {int(cut_f)}, ratio cut/(|V1||V2|) = {r2_f:.3f}")
    best = isoperimetric_number(nodes, A)[2]
    print(f"    (for comparison, the best possible ratio theta_G = {best[0]:.3f}"
          f" at V1 = {best[1]})")

    # (g) four clusters
    clusters, details = recursive_fiedler_4(nodes, A)
    print("\n(g) Recursive Fiedler: split each half again with its OWN Fiedler vector")
    for sub_nodes, l2, sv2, sneg, spos in details:
        print(f"    subgraph {sub_nodes}: lambda_2 = {l2:.3f}")
        print(f"       v_2 = {sv2}")
        print(f"       -> {sneg} | {spos}")
    print("    Four clusters:")
    for k, c in enumerate(clusters, 1):
        print(f"       C{k} ({len(c)} marmots): {c}")
    sizes = [len(c) for c in clusters]
    print(f"    sizes = {sizes}  (ideal for 15 marmots: about 4 each)")
    print(f"    total cut edges = {int(total_cut(nodes, A, clusters))}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    draw_clusters(axes[0], G, POS_MARMOT, [neg, pos],
                  heading=f"Ex 6(f): 2 clusters ($\\lambda_2$ = {lam2:.3f})")
    draw_fiedler_bars(axes[1], nodes, v2, heading="Ex 6(e): Fiedler vector")
    save(fig, "ex6_two_clusters.png")

    fig, ax = plt.subplots(figsize=(5.5, 5))
    draw_clusters(ax, G, POS_MARMOT, clusters,
                  heading="Ex 6(g): 4 clusters (recursive Fiedler)")
    save(fig, "ex6_four_clusters.png")
    return nodes, A, neg, pos, clusters


def exercise_7():
    title("EXERCISE 7 - More information: weighted and directed graphs")
    # ----- Part 1: number of interactions -> WEIGHTED graph ---------------
    # Real data: same 29 edges, Ex 6 (all weights 1) vs Ex 8 (real counts).
    nodes, A_plain = adjacency_matrix(build_graph(EDGES_EX6))
    _, A_weight = adjacency_matrix(build_graph(EDGES_EX8, weighted=True))
    degree = A_plain.sum(axis=1)       # number of friends
    strength = A_weight.sum(axis=1)    # number of interactions
    print("Part 1 - counting interactions turns 1's into weights w_ij.")
    print("   marmot : degree (unweighted) | strength (weighted)")
    for v, d, s in zip(nodes, degree, strength):
        print(f"   {v:>6} : {int(d):>6}              | {int(s):>6}")
    print(f"   Most connected by degree  : marmot {nodes[np.argmax(degree)]}")
    print(f"   Most connected by strength: marmot {nodes[np.argmax(strength)]}")
    print("   -> The 'most social' marmot changes once we count interactions.")

    # ----- Part 2: dominance -> DIRECTED graph -----------------------------
    print("\nPart 2 - dominance has a direction: i -> j means i dominates j.")
    print("   (Hypothetical 6-marmot example, made up for illustration only.)")
    D = nx.DiGraph()
    for i, j, w in EDGES_EX7_DOMINANCE:
        D.add_edge(i, j, weight=w)
    dnodes = sorted(D.nodes())
    Ad = nx.to_numpy_array(D, nodelist=dnodes, weight="weight")
    print_matrix("A_directed", Ad)
    print("   Is A symmetric?", np.allclose(Ad, Ad.T))

    out_w = Ad.sum(axis=1)     # times marmot i dominated others
    in_w = Ad.sum(axis=0)      # times marmot i was dominated
    print("   marmot : wins (out) | losses (in) | net score = out - in")
    for v, o, i in zip(dnodes, out_w, in_w):
        print(f"   {v:>6} : {int(o):>8}   | {int(i):>8}    | {int(o - i):>+5}")

    # A directed Laplacian L = D_out - A is NOT symmetric -> its eigenvalues
    # can be complex numbers, so "lambda_2" and the sign rule lose meaning.
    L_dir = np.diag(out_w) - Ad
    eig_dir = np.linalg.eigvals(L_dir)
    print("   eigenvalues of the directed Laplacian:", np.round(eig_dir, 3))
    print("   complex eigenvalues present?", np.any(np.abs(eig_dir.imag) > TOL))

    # To CLUSTER we can forget the direction: W = A + A^T (total interactions)
    W = Ad + Ad.T
    vals, lam2, v2 = fiedler(W)
    neg, pos = fiedler_split(dnodes, W)
    print("   Symmetrised W = A + A^T -> ordinary Fiedler method:")
    print(f"   lambda_2 = {lam2:.3f},  clusters {neg} | {pos}")
    print("   Summary: weights = HOW OFTEN, arrows = WHO dominates WHOM;")
    print("   a directed WEIGHTED graph (like this one) keeps both pieces of info.")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    posd = nx.circular_layout(D)
    ax = axes[0]
    nx.draw_networkx_nodes(D, posd, ax=ax, node_color="#8172B3", node_size=450)
    nx.draw_networkx_labels(D, posd, ax=ax, font_color="white", font_weight="bold")
    nx.draw_networkx_edges(D, posd, ax=ax, arrows=True, arrowsize=18,
                           connectionstyle="arc3,rad=0.12", node_size=450)
    nx.draw_networkx_edge_labels(D, posd, ax=ax,
                                 edge_labels=nx.get_edge_attributes(D, "weight"),
                                 font_size=8, connectionstyle="arc3,rad=0.12")
    ax.set_title("Ex 7: directed dominance graph (hypothetical)", fontsize=10)
    ax.axis("off")
    x = np.arange(len(nodes))
    axes[1].bar(x - 0.2, degree / degree.max(), width=0.4, label="degree (Ex 6)")
    axes[1].bar(x + 0.2, strength / strength.max(), width=0.4,
                label="strength (Ex 8)")
    axes[1].set_xticks(x, [str(v) for v in nodes])
    axes[1].set_xlabel("marmot")
    axes[1].set_ylabel("value / maximum")
    axes[1].set_title("Ex 7: unweighted vs weighted importance", fontsize=10)
    axes[1].legend()
    save(fig, "ex7_weighted_directed.png")


def exercise_8(ex6_neg, ex6_pos, ex6_clusters):
    title("EXERCISE 8 - Weighted marmot network (Figure 1.26)")
    G = build_graph(EDGES_EX8, weighted=True)
    nodes, A = adjacency_matrix(G)

    # (a) weighted Laplacian
    L = laplacian_matrix(A)
    print("(a) Weighted Laplacian: diagonal = total interactions of each marmot")
    print_matrix("L = D - W", L)

    # (b) Fiedler value / vector and two clusters
    vals, lam2, v2 = fiedler(A)
    neg, pos = fiedler_split(nodes, A)
    print(f"\n(b) Fiedler value lambda_2 = {lam2:.3f}"
          f"   (5 decimals {lam2:.5f} only to compare with book 8b: {BOOK_8B_LAMBDA2},"
          f" match: {abs(lam2 - BOOK_8B_LAMBDA2) < 1e-4})")
    print("    Fiedler vector v_2 =", v2)
    print("    V1 (v_2 < 0)  =", neg)
    print("    V2 (v_2 >= 0) =", pos)
    cut, _, r2 = isoperimetric_ratio(nodes, A, neg)
    cut_list = [(u, v, int(A[nodes.index(u), nodes.index(v)]))
                for u in neg for v in pos if A[nodes.index(u), nodes.index(v)] > 0]
    print(f"    cut edges (u, v, weight) = {cut_list}")
    print(f"    cut weight = {int(cut)} interactions,  cut/(|V1||V2|) = {r2:.3f}")

    # The vertex whose entry is closest to 0 is the most 'borderline' one.
    k = int(np.argmin(np.abs(v2)))
    b = nodes[k]
    print(f"    marmot {b} has the entry closest to 0 ({v2[k]:+.3f})"
          " -> it is the most 'borderline' marmot")
    alt = sorted(set(neg) ^ {b})          # move it to the other side
    cut_alt, _, r2_alt = isoperimetric_ratio(nodes, A, alt)
    print(f"    If marmot {b} switched side: cut weight = {int(cut_alt)}, "
          f"ratio = {r2_alt:.3f}")

    # Smallest isoperimetric ratio (Definition 2), brute force
    best = isoperimetric_number(nodes, A)
    print(f"    theta_G (Def. 2) = {best[2][0]:.3f} at V1 = {best[2][1]}")

    # Comparison with Exercise 6
    moved = sorted(set(neg) ^ set(ex6_neg))
    print("\nComparison with Ex 6 (unweighted):")
    print("    Ex 6:", ex6_neg, "|", ex6_pos)
    print("    Ex 8:", neg, "|", pos)
    print("    marmot(s) that changed side:", moved if moved else "none")

    # Erratum check: weights implied by the Laplacian of Sample Solution 8a
    A_8a = A.copy()
    for (u, v), w in BOOK_8A_WEIGHTS.items():
        A_8a[nodes.index(u), nodes.index(v)] = A_8a[nodes.index(v), nodes.index(u)] = w
    _, lam2_8a, _ = fiedler(A_8a)
    print("\nErratum check - Laplacian printed in Sample Solution 8a:")
    print("    differs from Figure 1.26 on edges", list(BOOK_8A_WEIGHTS))
    print(f"    it would give lambda_2 = {lam2_8a:.3f} and clusters",
          fiedler_split(nodes, A_8a), "-> NOT the book's answer 8b")

    # Extra: four weighted clusters, compared with Ex 6(g)
    clusters, _ = recursive_fiedler_4(nodes, A)
    print("\nExtra - four weighted clusters (same recursive method as Ex 6g):")
    for k, c in enumerate(clusters, 1):
        print(f"    C{k}: {c}")
    print("    Ex 6(g) gave:", ex6_clusters)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    draw_clusters(axes[0], G, POS_MARMOT, [neg, pos], show_weights=True,
                  heading=f"Ex 8: weighted, 2 clusters ($\\lambda_2$ = {lam2:.3f})")
    draw_clusters(axes[1], build_graph(EDGES_EX6), POS_MARMOT, [ex6_neg, ex6_pos],
                  heading="Ex 6 (unweighted) for comparison")
    save(fig, "ex8_weighted_vs_unweighted.png")


# ===========================================================================
# 5. MAIN PROGRAM
# ===========================================================================
if __name__ == "__main__":
    nodes4, A4, neg4, pos4 = exercise_4()
    exercise_5(nodes4, A4, neg4, pos4)
    nodes6, A6, neg6, pos6, clusters6 = exercise_6()
    exercise_7()
    exercise_8(neg6, pos6, clusters6)
    print("\nDone. All figures are in the 'figures' folder.")
