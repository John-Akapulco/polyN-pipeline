#!/usr/bin/env python3
"""
random_structure_generator.py
==============================

Random (non-exhaustive) generator of valid molecular graphs for a single
element at a fixed valence -- a scalable complement to MAYGEN's exhaustive
enumeration, which becomes intractable for large N (see N16 in this
project: 10+ hours without completing).

Algorithm (configuration-model stub-matching, standard in random graph
theory, adapted here to respect chemical valence):

1. Give every atom `valence` stubs (half-bonds) to satisfy.
2. Randomly pair stubs between DIFFERENT atoms (self-loops forbidden).
   Pairing the same pair of atoms more than once yields a double/triple bond.
3. If the resulting multigraph is disconnected (more than one molecular
   fragment), repeatedly repair it with random double-edge swaps between
   different components until it is a single connected molecule.
4. Build an RDKit molecule from the edge multiset, canonicalize the SMILES,
   and deduplicate across the batch.

This produces valid, connected, valence-correct random topologies at any
N in roughly O(N) per structure -- no combinatorial enumeration, so it
scales to arbitrarily large clusters where MAYGEN cannot finish in
practice.

Usage
-----
    python random_structure_generator.py -n 16 -k 200 -o N16_random.smi
    python random_structure_generator.py -n 16 -k 200 --valence 3 --element N

Output is a plain .smi file (one SMILES per line), directly compatible
with polyN_pipeline.py's input format.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from rdkit import Chem


def _stub_matching_multigraph(valence_seq: list, rng: random.Random,
                               max_attempts: int = 200, chain_seeded: bool = True):
    """Return a dict {(atom_i, atom_j): bond_order} for a random multigraph
    respecting the given PER-ATOM valence sequence (not necessarily
    uniform -- e.g. [1, 2, 2, ..., 2, 1] for a chain with degree-1 termini,
    or [4, 3, 3, ..., 3, 2] for a localized N+ hub / N- terminus motif),
    or None if max_attempts is exceeded without a valid self-loop-free
    pairing.

    If chain_seeded is True, a random Hamiltonian path is built first using
    up to 2 stubs per atom (capped by that atom's own valence -- so a
    valence-1 atom only ever gets 1 path stub, correctly staying terminal),
    and only the REMAINING stubs are paired randomly."""
    n = len(valence_seq)
    for _ in range(max_attempts):
        remaining = list(valence_seq)
        edges = {}

        if chain_seeded:
            order = list(range(n))
            rng.shuffle(order)
            for a, b in zip(order[:-1], order[1:]):
                if remaining[a] < 1 or remaining[b] < 1:
                    continue  # this atom's valence budget is already used up (e.g. a degree-1 terminus)
                key = (min(a, b), max(a, b))
                edges[key] = edges.get(key, 0) + 1
                remaining[a] -= 1
                remaining[b] -= 1

        stubs = [atom for atom in range(n) for _ in range(remaining[atom])]
        rng.shuffle(stubs)
        if len(stubs) % 2 != 0:
            continue
        pairs = list(zip(stubs[0::2], stubs[1::2]))
        if not all(a != b for a, b in pairs):
            continue

        for a, b in pairs:
            key = (min(a, b), max(a, b))
            edges[key] = edges.get(key, 0) + 1

        if all(order <= 3 for order in edges.values()):
            return edges
    return None


def _connected_components(n: int, edges: dict) -> list:
    adj = {i: set() for i in range(n)}
    for (a, b) in edges:
        adj[a].add(b)
        adj[b].add(a)
    seen = set()
    components = []
    for start in range(n):
        if start in seen:
            continue
        stack, comp = [start], set()
        while stack:
            u = stack.pop()
            if u in comp:
                continue
            comp.add(u)
            stack.extend(adj[u] - comp)
        seen |= comp
        components.append(comp)
    return components


def _repair_connectivity(n: int, edges: dict, rng: random.Random,
                          max_swaps: int = 500) -> dict:
    """Reconnect a disconnected multigraph with random double-edge swaps
    (standard degree-sequence-preserving rewiring): pick one edge from each
    of two different components and swap endpoints so a bond now bridges
    them. Preserves every atom's valence exactly."""
    edges = dict(edges)
    for _ in range(max_swaps):
        components = _connected_components(n, edges)
        if len(components) == 1:
            return edges
        c1, c2 = rng.sample(components, 2)
        edges_c1 = [e for e in edges if e[0] in c1 or e[1] in c1]
        edges_c2 = [e for e in edges if e[0] in c2 or e[1] in c2]
        if not edges_c1 or not edges_c2:
            continue
        (a, b) = rng.choice(edges_c1)
        (c, d) = rng.choice(edges_c2)
        if len({a, b, c, d}) < 4:
            continue
        # swap: remove (a,b) and (c,d), add (a,c) and (b,d) -- bridges the
        # two components while every atom keeps the same total valence
        for e in [(a, b), (c, d)]:
            key = (min(e), max(e))
            edges[key] -= 1
            if edges[key] == 0:
                del edges[key]
        for e in [(a, c), (b, d)]:
            key = (min(e), max(e))
            edges[key] = edges.get(key, 0) + 1
    return edges  # may still be disconnected if max_swaps exhausted


_STANDARD_VALENCE = {"N": 3, "P": 3, "As": 3, "O": 2, "S": 2, "Se": 2, "H": 1,
                     "F": 1, "Cl": 1, "Br": 1, "I": 1, "C": 4, "Si": 4}


def _random_valence_sequence(n: int, target_charge: int, rng: random.Random,
                              defect_valences: tuple = (2, 4), standard_valence: int = 3,
                              max_extra_pairs: int = 1) -> list:
    """Randomly build a per-atom valence sequence for n atoms of standard
    valence `standard_valence` (3 for N), such that the sum of formal
    charges equals target_charge -- i.e. a genuine random MIXTURE of local
    coordination environments (e.g. terminal/azide-like dicoordinate anion
    centers at valence 2, normal tricoordinate backbone at valence 3,
    ammonium-like tetracoordinate cation centers at valence 4), with both
    the NUMBER and the PLACEMENT of defect centers randomized from one
    sample to the next.

    `defect_valences` gives the (low, high) valences available besides the
    standard one -- by default (2, 4) for N, corresponding to formal
    charges -1 and +1 respectively. `max_extra_pairs` allows additional
    balanced (+1, -1) defect pairs beyond the minimum required to reach
    target_charge, for structural diversity (e.g. a net +1 cation that
    ALSO happens to carry one azide-like terminus and one extra ammonium-
    like hub, still net +1 overall)."""
    low_val, high_val = defect_valences
    low_charge, high_charge = low_val - standard_valence, high_val - standard_valence  # e.g. -1, +1

    total_valence_needed = standard_valence * n + target_charge
    if total_valence_needed % 2 != 0:
        raise ValueError(
            f"No valid structure exists: {n} atoms at standard valence {standard_valence} "
            f"with net charge {target_charge} requires an odd total valence sum "
            f"({total_valence_needed}) -- violates the handshake lemma."
        )

    n_plus_min = max(0, target_charge // high_charge) if target_charge > 0 else 0
    n_minus_min = max(0, (-target_charge) // (-low_charge)) if target_charge < 0 else 0
    # (high_charge is +1 and low_charge is -1 by default, so these are just
    # |target_charge| in the common case; kept general for other menus.)

    n_extra = rng.randint(0, max_extra_pairs)
    n_plus = n_plus_min + n_extra
    n_minus = n_minus_min + n_extra
    if n_plus + n_minus > n:
        n_plus, n_minus = n_plus_min, n_minus_min  # fall back to the minimum if too crowded
    if n_plus + n_minus > n:
        raise ValueError(f"Cannot place {n_plus + n_minus} defect centers among only {n} atoms.")

    positions = list(range(n))
    rng.shuffle(positions)
    valence_seq = [standard_valence] * n
    for i in positions[:n_plus]:
        valence_seq[i] = high_val
    for i in positions[n_plus:n_plus + n_minus]:
        valence_seq[i] = low_val

    return valence_seq


def _edges_to_smiles(n: int, edges: dict, symbol: str, valence_seq: list = None,
                      symmetric_charge: bool = False) -> str | None:
    mol = Chem.RWMol()
    std_valence = _STANDARD_VALENCE.get(symbol)
    for i in range(n):
        atom = Chem.Atom(symbol)
        # CRITICAL: never let RDKit silently complete an under-specified
        # valence with an implicit hydrogen. Without this, an atom left
        # deliberately under-coordinated (e.g. a valence-2 anion-terminus
        # center with no formal charge set below) would get a REAL,
        # UNINTENDED hydrogen atom added during sanitization -- exactly the
        # contamination bug found in production (H appearing in "pure" Nn
        # anion structures). The molecule's connectivity here is always
        # fully and deliberately specified by the stub-matching algorithm,
        # so implicit valence completion is never wanted, charged or not.
        atom.SetNoImplicit(True)
        if valence_seq is not None and std_valence is not None:
            excess = valence_seq[i] - std_valence
            # Excess valence (e.g. a valence-4 ammonium-like hub) ALWAYS
            # needs an explicit formal charge, or RDKit rejects the atom
            # outright. Deficient valence (e.g. a valence-2 azide/
            # pentazolate-like center) is more ambiguous: it could mean a
            # neutral radical (no formal charge -- the default, used for
            # plain topology "starting guesses" whose real net charge is
            # applied separately at the xtb level) or a genuine closed-shell
            # anionic center (explicit negative formal charge, needed when
            # deliberately constructing a specific target-net-charge Lewis
            # structure). symmetric_charge selects the latter. Either way,
            # SetNoImplicit above guarantees no spurious H is added regardless
            # of which branch is taken here.
            if excess > 0 or (symmetric_charge and excess < 0):
                atom.SetFormalCharge(excess)
        mol.AddAtom(atom)
    order_map = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}
    for (a, b), order in edges.items():
        mol.AddBond(a, b, order_map[order])
    try:
        m = mol.GetMol()
        Chem.SanitizeMol(m)
        return Chem.MolToSmiles(m)
    except Exception:
        return None


def generate_random_structures(n: int, k: int, valence=3, element: str = "N",
                                seed: int = None, max_tries_factor: int = 20,
                                chain_seeded: bool = True, target_charge: int = None,
                                defect_valences: tuple = (2, 4), max_extra_pairs: int = 1) -> list:
    """Generate up to k distinct, connected, valence-correct random SMILES
    for a cluster of n atoms of `element`.

    Two modes:
      - Fixed valence (default): `valence` is a single int (uniform) or an
        explicit per-atom list -- the SAME valence pattern is used for
        every sampled structure, only the connectivity is randomized.
      - Random valence mode (target_charge is not None): for EACH sampled
        structure, a fresh random valence sequence is drawn (see
        _random_valence_sequence) placing terminal/hub defect centers at
        random positions and in random numbers (within max_extra_pairs),
        such that the formal charges always sum to target_charge. This is
        the genuinely stochastic mixture of coordination environments
        (terminal, di-, tri-, tetracoordinate) requested for exploring
        e.g. azide-like or ammonium-like motifs at random positions.
    """
    rng = random.Random(seed)
    std_valence = _STANDARD_VALENCE.get(element, 3)

    if target_charge is None:
        if isinstance(valence, int):
            valence_seq = [valence] * n
        else:
            valence_seq = list(valence)
            if len(valence_seq) != n:
                raise ValueError(f"valence sequence has {len(valence_seq)} entries, expected {n}")
        if sum(valence_seq) % 2 != 0:
            raise ValueError(f"sum of valences ({sum(valence_seq)}) is odd -- "
                              f"no valid molecular graph exists (handshake lemma).")

    smiles_seen = set()
    max_tries = k * max_tries_factor

    tries = 0
    while len(smiles_seen) < k and tries < max_tries:
        tries += 1
        if target_charge is not None:
            valence_seq = _random_valence_sequence(
                n, target_charge, rng, defect_valences=defect_valences,
                standard_valence=std_valence, max_extra_pairs=max_extra_pairs)

        edges = _stub_matching_multigraph(valence_seq, rng, chain_seeded=chain_seeded)
        if edges is None:
            continue
        components = _connected_components(n, edges)
        if len(components) > 1:
            edges = _repair_connectivity(n, edges, rng)
            if len(_connected_components(n, edges)) > 1:
                continue  # repair failed this attempt, discard and retry
        smi = _edges_to_smiles(n, edges, element, valence_seq,
                                symmetric_charge=(target_charge is not None))
        if smi is not None:
            smiles_seen.add(smi)

    return sorted(smiles_seen)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-n", "--n-atoms", type=int, required=True,
                         help="Number of atoms in the cluster")
    parser.add_argument("-k", "--n-structures", type=int, default=100,
                         help="Number of distinct random structures to generate (default: 100)")
    parser.add_argument("--valence", type=int, default=3,
                         help="Fixed valence per atom, used unless --valence-pattern is given "
                              "(default: 3, neutral trivalent N)")
    parser.add_argument("--valence-pattern", type=str, default=None,
                         help="Explicit per-atom valence sequence, comma-separated (length must "
                              "equal --n-atoms), e.g. '1,2,2,2,2,1' for an open chain with "
                              "degree-1 termini, or '4,3,3,3,2' for a localized N+ hub "
                              "(valence 4) and an N- terminus (valence 2) amid a valence-3 "
                              "backbone. Overrides --valence when given.")
    parser.add_argument("--element", type=str, default="N",
                         help="Element symbol (default: N)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: unseeded)")
    parser.add_argument("--target-charge", type=int, default=None,
                         help="Enable random-valence mode: for each sampled structure, "
                              "randomly place terminal (-1, valence 2) and/or hub (+1, "
                              "valence 4) defect centers among an otherwise normal valence-3 "
                              "backbone, such that formal charges sum to this net charge. "
                              "Overrides --valence/--valence-pattern when given.")
    parser.add_argument("--defect-valences", type=str, default="2,4",
                         help="Comma-separated (low, high) defect valences for --target-charge "
                              "mode (default: '2,4', i.e. -1/+1 formal charge for N)")
    parser.add_argument("--extra-pairs-max", type=int, default=1,
                         help="Max additional balanced (+1,-1) defect pairs beyond the minimum "
                              "needed to reach --target-charge, for structural diversity "
                              "(default: 1)")
    parser.add_argument("--no-chain-seed", action="store_true",
                         help="Disable chain-seeding (pure uniform random stub-matching; "
                              "produces more densely fused, harder-to-embed topologies)")
    parser.add_argument("-o", "--output", type=str, required=True,
                         help="Output DIRECTORY for N<n>_<family>_from_random.smi "
                              "files (classified by net charge, like the other "
                              "generators). For backward compatibility, if this "
                              "ends in '.smi' it is treated as a single output file.")
    args = parser.parse_args()

    if args.target_charge is not None:
        low, high = (int(v) for v in args.defect_valences.split(","))
        smiles = generate_random_structures(
            args.n_atoms, args.n_structures, element=args.element, seed=args.seed,
            chain_seeded=not args.no_chain_seed, target_charge=args.target_charge,
            defect_valences=(low, high), max_extra_pairs=args.extra_pairs_max,
        )
    else:
        if args.valence_pattern:
            valence = [int(v) for v in args.valence_pattern.split(",")]
            if len(valence) != args.n_atoms:
                raise SystemExit(f"ERROR: --valence-pattern has {len(valence)} entries, "
                                  f"expected {args.n_atoms} (--n-atoms).")
        else:
            valence = args.valence

        valence_seq = valence if isinstance(valence, list) else [valence] * args.n_atoms
        if sum(valence_seq) % 2 != 0:
            raise SystemExit(
                f"ERROR: sum of valences = {sum(valence_seq)} is odd -- "
                f"no valid molecular graph exists (handshake lemma)."
            )

        smiles = generate_random_structures(
            args.n_atoms, args.n_structures, valence=valence,
            element=args.element, seed=args.seed, chain_seeded=not args.no_chain_seed,
        )

    out = Path(args.output)

    # Backward-compatible single-file mode if the user passed a .smi path.
    if out.suffix == ".smi":
        out.write_text("\n".join(smiles) + "\n")
        print(f"Wrote {len(smiles)} / {args.n_structures} requested distinct "
              f"structures to {out}")
    else:
        # Directory mode (harmonized with cxhx_to_nx.py and geng_enumerate.py):
        # classify each SMILES by its ACTUAL RDKit net charge and write
        # N<n>_<family>_from_random.smi files. This lets the pipeline read the
        # output uniformly, and keeps only |charge| <= 1 species.
        from rdkit import Chem
        from rdkit import RDLogger
        from collections import defaultdict
        RDLogger.DisableLog("rdApp.*")

        def family_of(q):
            return "neutral" if q == 0 else ("cation" if q > 0 else "anion")

        out.mkdir(parents=True, exist_ok=True)
        buckets = defaultdict(list)
        n_rejected = 0
        for smi in smiles:
            m = Chem.MolFromSmiles(smi)
            if m is None:
                continue
            q = Chem.GetFormalCharge(m)
            if abs(q) > 1:              # keep only neutral / +1 / -1
                n_rejected += 1
                continue
            buckets[family_of(q)].append(smi)

        total = 0
        for family, smis in sorted(buckets.items()):
            fpath = out / f"N{args.n_atoms}_{family}_from_random.smi"
            with open(fpath, "a") as fh:   # append: accumulate across calls
                for smi in smis:
                    fh.write(smi + "\n")
            total += len(smis)
            print(f"N{args.n_atoms} [{family}]: {len(smis)} structure(s) -> {fpath}")
        print(f"[random] wrote {total} structure(s) to {out}"
              + (f" ({n_rejected} rejected for |charge|>1)" if n_rejected else ""))

    if len(smiles) < args.n_structures:
        print("(fewer than requested: the random search saturated -- likely near the "
              "true isomer count for this size, or max_tries_factor needs raising)")


if __name__ == "__main__":
    main()
