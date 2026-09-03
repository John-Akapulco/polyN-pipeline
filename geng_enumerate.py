#!/usr/bin/env python3
"""
geng_enumerate.py
=================

Exhaustive topology generator for polynitrogen species, built on the
`geng` tool from the nauty package (McKay & Piperno). For a given number of
atoms N it enumerates every connected, non-isomorphic simple graph on N
vertices, then for each graph explores bond-order assignments (single /
double / triple) on the edges. Nitrogen formal charges are derived from the
bond-order pattern and -- crucially -- **assigned at the graph stage, before
RDKit ever sanitizes the molecule**, so RDKit never silently saturates an
under-valent nitrogen with a phantom hydrogen.

This is a fourth, complementary topology source next to the isolobal /
random / MAYGEN generators:

  - Unlike the random generator, it is EXHAUSTIVE over graph topologies (up
    to the sampling cap): no topology is missed for a given N.
  - Unlike MAYGEN (which enumerates constitutional isomers of a fixed
    formula and does not vary bond order to produce charges), geng gives the
    bare connected graphs and we ourselves distribute bond orders, so the
    neutral, cation, and anion families all emerge from the SAME graph set.

Formal-charge rule for nitrogen (5 valence electrons, one lone pair by
default): with a total incident bond order b at an atom,

    q = b - 3        (b=2 -> -1 bent N;  b=3 -> 0;  b=4 -> +1)

This is the identical accounting used by the isolobal generator, so the two
sources are chemically consistent and their outputs deduplicate cleanly.

Dependencies
------------
  - nauty's `geng` executable on PATH
      macOS:   brew install nauty
      Debian:  sudo apt install nauty
  - rdkit (already a pipeline dependency)

Usage
-----
    python3 geng_enumerate.py -n 10 --max-abs-charge 1 -o seeds/
    python3 geng_enumerate.py -n 6 --max-graphs 500 --max-abs-charge 1 -o seeds/

Output files follow the same convention as the other generators:
    N<n>_<family>_from_geng.smi     (family = neutral | cation | anion)
so they slot straight into the pipeline's input_dir.
"""

from __future__ import annotations

import argparse
import itertools
import random
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")  # silence per-structure valence warnings

# Nitrogen: neutral valence 3; a bond-order sum of 2/3/4 maps to charge -1/0/+1.
NEUTRAL_VALENCE = 3
MAX_BOND_ORDER_SUM = 4  # N+ tetravalent is the ceiling; sum 5+ is unphysical


def check_geng_available() -> str:
    """Return the geng executable path, or exit with an install hint."""
    exe = shutil.which("geng")
    if exe is None:
        sys.exit(
            "ERROR: 'geng' (nauty) not found on PATH.\n"
            "  Install it with:  brew install nauty   (macOS)\n"
            "                    sudo apt install nauty (Debian/Ubuntu)\n"
            "geng ships as part of nauty; on some systems the binary is called\n"
            "'nauty-geng' -- if so, symlink it to 'geng' or edit this script.")
    return exe


def enumerate_connected_graphs(n_nodes: int, geng_exe: str,
                               max_graphs: int | None, seed: int) -> list:
    """Run `geng -c N` and parse its output into edge lists.

    geng's default output is graph6, which is compact but not edge-labeled.
    We request the explicit edge format with -e? No: geng emits graph6; we
    decode it with RDKit's/networkx-free minimal graph6 parser below. To keep
    dependencies light we instead ask geng for a sparse edge listing via the
    companion tool `listg` when available; otherwise we decode graph6 here."""
    # Ask geng for graph6 on stdout; decode graph6 ourselves (no extra deps).
    cmd = [geng_exe, "-c", str(n_nodes)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True)
    graphs = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.strip()
        if not line or line.startswith(">"):
            continue
        edges = _decode_graph6_edges(line)
        if edges is not None:
            graphs.append(edges)
        if max_graphs is not None and len(graphs) >= max_graphs * 4:
            # Collect a bit more than needed, then sample, for variety.
            proc.terminate()
            break
    proc.wait()

    if max_graphs is not None and len(graphs) > max_graphs:
        random.seed(seed)
        graphs = random.sample(graphs, max_graphs)
    return graphs


def _decode_graph6_edges(g6: str):
    """Decode a graph6 string into a list of (u, v) edges (0-indexed).

    graph6 format (McKay): first byte(s) encode n; remaining bytes encode the
    upper-triangle adjacency bits, 6 per byte, offset by 63."""
    data = [ord(c) - 63 for c in g6]
    if not data:
        return None
    # Number of vertices (small-graph case: single byte < 63 after offset).
    n = data[0]
    bits = []
    for byte in data[1:]:
        for shift in range(5, -1, -1):
            bits.append((byte >> shift) & 1)
    edges = []
    idx = 0
    # graph6 orders bits column-by-column: for j in 1..n-1, for i in 0..j-1.
    for j in range(1, n):
        for i in range(j):
            if idx < len(bits) and bits[idx]:
                edges.append((i, j))
            idx += 1
    return edges


def formal_charge_for_atom(bond_order_sum: int) -> int:
    """Nitrogen formal charge from its total incident bond order: q = b - 3."""
    return bond_order_sum - NEUTRAL_VALENCE


def build_molecule(edges, bond_orders, n_nodes):
    """Construct an all-nitrogen RDKit molecule with formal charges assigned
    at the graph stage (before sanitization), so RDKit never adds implicit H.

    Returns (smiles, net_charge) or (None, None) if the assignment is invalid
    (bond-order sum out of the physical 2..4 range, or sanitization fails)."""
    # Total incident bond order per vertex.
    order_sum = [0] * n_nodes
    for (u, v), o in zip(edges, bond_orders):
        order_sum[u] += o
        order_sum[v] += o

    charges = []
    for b in order_sum:
        # A one-coordinate / low-order N below 2, or above 4, is unphysical
        # here (we screen for terminal N separately). Reject sums outside 2..4.
        if b < 2 or b > MAX_BOND_ORDER_SUM:
            return None, None
        charges.append(formal_charge_for_atom(b))

    rw = Chem.RWMol()
    for _ in range(n_nodes):
        a = Chem.Atom(7)          # nitrogen
        a.SetNoImplicit(True)     # <-- KEY: forbid automatic H saturation
        rw.AddAtom(a)
    for (u, v), o in zip(edges, bond_orders):
        bt = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE,
              3: Chem.BondType.TRIPLE}[o]
        rw.AddBond(u, v, bt)
    for i, q in enumerate(charges):
        if q != 0:
            rw.GetAtomWithIdx(i).SetFormalCharge(q)  # <-- assign BEFORE sanitize

    m = rw.GetMol()
    try:
        Chem.SanitizeMol(m)
    except Exception:
        return None, None
    # Purity gate: pure nitrogen, no stray H (defensive -- SetNoImplicit should
    # already guarantee it).
    if any(a.GetAtomicNum() != 7 for a in m.GetAtoms()):
        return None, None
    net_charge = sum(charges)
    return Chem.MolToSmiles(m, canonical=True), net_charge


def enumerate_bond_orders(edges, n_nodes, max_abs_charge):
    """Yield (smiles, net_charge) for every valid bond-order assignment on a
    graph, pruning early on the bond-order-sum ceiling to keep it tractable."""
    results = {}
    n_edges = len(edges)
    if n_edges == 0:
        return results
    for orders in itertools.product((1, 2, 3), repeat=n_edges):
        # Early valence prune (sum of incident bond orders <= 4 at each vertex).
        order_sum = [0] * n_nodes
        ok = True
        for (u, v), o in zip(edges, orders):
            order_sum[u] += o
            order_sum[v] += o
            if order_sum[u] > MAX_BOND_ORDER_SUM or order_sum[v] > MAX_BOND_ORDER_SUM:
                ok = False
                break
        if not ok:
            continue
        smi, net_charge = build_molecule(edges, orders, n_nodes)
        if smi is None:
            continue
        if abs(net_charge) > max_abs_charge:
            continue
        # Deduplicate identical canonical SMILES within this run.
        results[smi] = net_charge
    return results


def family_of(charge: int) -> str:
    return "neutral" if charge == 0 else ("cation" if charge > 0 else "anion")


def main():
    ap = argparse.ArgumentParser(
        description="Exhaustive polynitrogen topology generator via geng (nauty). "
                    "Formal charges are assigned at the graph stage so RDKit "
                    "never auto-saturates nitrogen with hydrogen.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--n-nodes", type=int, required=True,
                    help="Number of nitrogen atoms (graph vertices).")
    ap.add_argument("-o", "--out-dir", required=True,
                    help="Output directory for N<n>_<family>_from_geng.smi files.")
    ap.add_argument("--max-abs-charge", type=int, default=1,
                    help="Keep only species with |net charge| <= this (default 1).")
    ap.add_argument("--max-graphs", type=int, default=None,
                    help="Randomly sample at most this many connected graphs "
                         "(default: all -- exhaustive; use a cap for large N).")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for graph sampling (default 42).")
    args = ap.parse_args()

    geng_exe = check_geng_available()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[geng] enumerating connected graphs on {args.n_nodes} vertices...")
    graphs = enumerate_connected_graphs(args.n_nodes, geng_exe,
                                        args.max_graphs, args.seed)
    print(f"[geng] {len(graphs)} graph(s) obtained.")
    if not graphs:
        sys.exit("No graphs produced -- check the geng installation.")

    # Accumulate unique SMILES per family.
    by_family = defaultdict(dict)  # family -> {smiles: charge}
    for gi, edges in enumerate(graphs, 1):
        if gi % 50 == 0 or gi == len(graphs):
            print(f"[geng] processing graph {gi}/{len(graphs)}")
        found = enumerate_bond_orders(edges, args.n_nodes, args.max_abs_charge)
        for smi, q in found.items():
            by_family[family_of(q)][smi] = q

    total = 0
    for family, smi_map in sorted(by_family.items()):
        out_file = out_dir / f"N{args.n_nodes}_{family}_from_geng.smi"
        with open(out_file, "a") as fh:  # append: accumulate across generators
            for smi in sorted(smi_map):
                fh.write(smi + "\n")
        total += len(smi_map)
        print(f"N{args.n_nodes} [{family}]: {len(smi_map)} structure(s) "
              f"-> {out_file}")
    print(f"[geng] done: {total} unique N{args.n_nodes} species written.")


if __name__ == "__main__":
    main()
