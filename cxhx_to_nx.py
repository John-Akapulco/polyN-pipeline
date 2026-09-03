#!/usr/bin/env python3
"""
cxhx_to_nx.py
=============

Isolobal CH -> N substitution: converts known CnHn hydrocarbon topologies
(polyhedranes, annulenes, and their valence isomers -- tetrahedrane,
prismanes, cubane, benzvalene, Dewar benzene, etc.) into candidate Nn
polyazote topologies.

Electronic justification: a CH framework vertex uses its 4 valence
electrons as 3 framework (C-C) bonds + 1 C-H bonding pair. A neutral N
atom has 5 valence electrons: 3 for the same framework bonds + 1 LONE PAIR
in place of the C-H pair. The substitution is therefore isoelectronic on
the sigma framework, not just a shape analogy.

Validated against polyN_pipeline's own MAYGEN-derived results: substituting
tetrahedrane (C4H4) reproduces EXACTLY the tetrahedral N4 isomer
independently found by MAYGEN's combinatorial enumeration.

Scope and limitations
----------------------
- Only applies to CnHn compounds where EVERY carbon has exactly one H and
  three heavy-atom (framework) neighbours -- NOT to all-carbon cages
  without exocyclic H (e.g. fullerenes), where each carbon instead
  contributes to a delocalised pi system with no lone pair to spare.
- By the same handshake-lemma parity argument used elsewhere in this
  project, a neutral CnHn compound can only be closed-shell for EVEN n
  (odd-n "annulenyl" species, e.g. C5H5, are well-known radicals) --
  exactly mirroring why neutral Nn has no closed-shell form for odd n.
- This method only yields topologies that correspond to an ALREADY-KNOWN,
  catalogued hydrocarbon scaffold -- a much smaller, non-exhaustive set
  compared to combinatorial (MAYGEN) or random (random_structure_generator.py)
  generation. It is a complementary, chemically-informed SEED source, not
  a replacement for broader exploration.

Usage
-----
    python cxhx_to_nx.py --list                     # show built-in library
    python cxhx_to_nx.py --all -o seeds/             # convert everything, one .smi per N-size
    python cxhx_to_nx.py --smiles "C1=CC=CC=C1"       # convert a single custom CnHn SMILES
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

# Verified canonical SMILES (PubChem), all CnHn cage/ring hydrocarbons and
# valence isomers where every carbon carries exactly one hydrogen.
KNOWN_CNHN = {
    # --- C4H4 ---
    "cyclobutadiene": "C1=CC=C1",
    "tetrahedrane": "C12C3C1C23",
    # --- C6H6 (benzene and three of its well-known valence isomers) ---
    "benzene": "C1=CC=CC=C1",
    "dewar_benzene": "C1=CC2C1C=C2",
    "benzvalene": "C1=CC2C3C1C23",
    "prismane": "C12C3C1C4C2C34",
    # --- C8H8 ---
    "cubane": "C12C3C4C1C5C2C3C45",
    "cyclooctatetraene": "C1=CC=CC=CC=C1",
}


def build_prismane_cxhx(n_gon: int) -> str:
    """Systematically build the [n]prismane graph: two parallel n-gon rings
    connected by n 'vertical' single bonds (every vertex trivalent, all
    single bonds). Verified against the known PubChem prismane structure
    for n=3; [4]prismane is graph-identical to cubane (both are the cube
    graph) -- a useful internal consistency check.

    Some larger members of this family (n>=6) may not correspond to an
    experimentally realised hydrocarbon (per the literature, hexaprismane
    C12H12 and beyond are computationally studied but not yet all
    synthesised) -- they are still legitimate, well-defined Lewis
    structures / candidate topologies for this purpose."""
    mol = Chem.RWMol()
    for _ in range(2 * n_gon):
        mol.AddAtom(Chem.Atom(6))
    for i in range(n_gon):
        mol.AddBond(i, (i + 1) % n_gon, Chem.BondType.SINGLE)
        mol.AddBond(n_gon + i, n_gon + (i + 1) % n_gon, Chem.BondType.SINGLE)
        mol.AddBond(i, n_gon + i, Chem.BondType.SINGLE)
    m = mol.GetMol()
    Chem.SanitizeMol(m)
    return Chem.MolToSmiles(m)


# [4]prismane is included even though it duplicates cubane's graph, for
# completeness of the systematic series; [5..8]prismane reach N10-N16.
PRISMANE_FAMILY = {f"prismane_{n}": build_prismane_cxhx(n) for n in range(3, 9)}


def cxhx_to_nx(smiles_cxhy: str) -> tuple:
    """General isolobal C -> N substitution for ANY hydrocarbon (not just
    CnHn where every carbon has exactly one H). Each carbon is replaced by
    N and its hydrogens are dropped; the resulting formal charge is:

        charge = (framework bond order, i.e. bond order sum to non-H
                  neighbours only) - 3

    since a neutral N has 3 valence bonds available where a neutral C has
    4 (3 framework + 1 C-H pair). This charge is not imposed -- it EMERGES
    from the precursor's bonding pattern, so scanning a broad hydrocarbon
    database and grouping results by (n_atoms, net_charge) directly
    populates the neutral/cation/anion families used throughout this
    project. Validated: cyclopentadiene -> pentazolate (N5-, charge -1,
    exactly as expected); tetrahedrane/cyclobutadiene reproduce the two
    known MAYGEN N4 isomers exactly (charge 0 in both cases, as expected
    for CnHn precursors where every C has exactly one H).

    Returns (nx_smiles, net_charge, error)."""
    mol = Chem.MolFromSmiles(smiles_cxhy)
    if mol is None:
        return None, None, "invalid source SMILES"

    mol = Chem.AddHs(mol)
    rw = Chem.RWMol(mol)
    h_to_remove = []
    net_charge = 0

    for atom in rw.GetAtoms():
        if atom.GetSymbol() == "H":
            continue
        if atom.GetSymbol() != "C":
            return None, None, f"non-carbon atom found ({atom.GetSymbol()}) -- not a pure hydrocarbon"
        h_neighbors = [nb for nb in atom.GetNeighbors() if nb.GetSymbol() == "H"]
        framework_bond_order = sum(
            b.GetBondTypeAsDouble() for b in atom.GetBonds()
            if b.GetOtherAtom(atom).GetSymbol() != "H"
        )
        charge = int(round(framework_bond_order)) - 3
        atom.SetAtomicNum(7)  # C -> N
        atom.SetNoImplicit(True)  # never let RDKit re-add an implicit H to
                                   # complete an under-valent N (same silent-
                                   # hydrogen trap fixed in the generator)
        if charge != 0:
            atom.SetFormalCharge(charge)
        net_charge += charge
        h_to_remove.extend(nb.GetIdx() for nb in h_neighbors)

    for idx in sorted(h_to_remove, reverse=True):
        rw.RemoveAtom(idx)

    try:
        m = rw.GetMol()
        Chem.SanitizeMol(m)
        result_smiles = Chem.MolToSmiles(m)
        # Final purity gate: after C->N substitution and H removal, the
        # result must be a pure nitrogen species. If ANY hydrogen survived
        # (e.g. an orphan explicit H that RDKit's sanitizer refused to drop,
        # the "not removing hydrogen atom without neighbors" case), reject
        # the structure rather than emit a contaminated N_xH_y. This is the
        # minimal correct criterion -- it keeps every genuinely clean
        # conversion (cyclopentadiene -> pentazolate included) while
        # discarding anything that would introduce a foreign atom.
        check = Chem.AddHs(Chem.MolFromSmiles(result_smiles))
        if any(a.GetSymbol() == "H" for a in check.GetAtoms()):
            return None, None, "hydrogen survived substitution -- contaminated result rejected"
        return result_smiles, net_charge, None
    except Exception as exc:
        return None, None, str(exc)


def n_size(nx_smiles: str) -> int:
    mol = Chem.MolFromSmiles(nx_smiles)
    return mol.GetNumAtoms()


def charge_family(charge: int) -> str:
    if charge < 0:
        return "anion"
    if charge > 0:
        return "cation"
    return "neutral"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="List the built-in CnHn library and exit")
    parser.add_argument("--all", action="store_true",
                         help="Convert every compound in the built-in library")
    parser.add_argument("--name", type=str, default=None,
                         help="Convert a single named compound from the built-in library")
    parser.add_argument("--smiles", type=str, default=None,
                         help="Convert a single custom hydrocarbon SMILES")
    parser.add_argument("--smiles-file", type=str, default=None,
                         help="Bulk-scan a file of hydrocarbon SMILES (one per line, e.g. exported "
                              "from PubChemPy/ChEMBL/ZINC by formula CxHy) -- the resulting net "
                              "charge is computed automatically for each, so this is the practical "
                              "way to mine a large hydrocarbon database for candidates at every "
                              "(n_atoms, charge_family) combination in one pass")
    parser.add_argument("--max-abs-charge", type=int, default=2,
                         help="Skip results whose net charge exceeds this in absolute value "
                              "(default: 2) -- guards against chemically implausible results where "
                              "charge piles up on a single under-coordinated carbon (e.g. a lone "
                              "terminal CH3 group gives charge -2 on that one atom already; several "
                              "such groups in one precursor can give very high, unrealistic net "
                              "charges). Set to a large number to disable.")
    parser.add_argument("-o", "--output-dir", type=str, default=".",
                         help="Output directory; one <N-size>_<family>_from_cxhx.smi file per "
                              "(size, family) combination (appended to, not overwritten)")
    args = parser.parse_args()

    if args.list:
        print("-- Bibliotheque nommee (composes reels caracterises ou historiquement proposes) --")
        for name, smi in KNOWN_CNHN.items():
            mol = Chem.MolFromSmiles(smi)
            formula = rdMolDescriptors.CalcMolFormula(mol) if mol else "?"
            print(f"  {name:20s} {formula:8s} {smi}")
        print("-- Famille systematique des [n]prismanes (verifiee n=3 vs PubChem) --")
        for name, smi in PRISMANE_FAMILY.items():
            mol = Chem.MolFromSmiles(smi)
            formula = rdMolDescriptors.CalcMolFormula(mol) if mol else "?"
            print(f"  {name:20s} {formula:8s} {smi}")
        return

    results = []  # (name, nx_smiles, net_charge)
    if args.all:
        for name, smi in {**KNOWN_CNHN, **PRISMANE_FAMILY}.items():
            nx_smi, charge, err = cxhx_to_nx(smi)
            if err:
                print(f"[SKIP] {name}: {err}")
            else:
                results.append((name, nx_smi, charge))
    elif args.name:
        library = {**KNOWN_CNHN, **PRISMANE_FAMILY}
        if args.name not in library:
            raise SystemExit(f"Unknown compound '{args.name}'. Use --list to see available names.")
        nx_smi, charge, err = cxhx_to_nx(library[args.name])
        if err:
            raise SystemExit(f"Conversion failed: {err}")
        results.append((args.name, nx_smi, charge))
    elif args.smiles:
        nx_smi, charge, err = cxhx_to_nx(args.smiles)
        if err:
            raise SystemExit(f"Conversion failed: {err}")
        results.append(("custom", nx_smi, charge))
    elif args.smiles_file:
        n_ok, n_skip = 0, 0
        with open(args.smiles_file) as fh:
            for line_no, line in enumerate(fh, 1):
                smi = line.strip().split()[0] if line.strip() else ""
                if not smi or smi.startswith("#"):
                    continue
                nx_smi, charge, err = cxhx_to_nx(smi)
                if err:
                    n_skip += 1
                    continue
                if abs(charge) > args.max_abs_charge:
                    n_skip += 1
                    continue
                results.append((f"line{line_no}", nx_smi, charge))
                n_ok += 1
        print(f"Scanned {args.smiles_file}: {n_ok} usable, {n_skip} skipped "
              f"(invalid / non-hydrocarbon / charge beyond +/-{args.max_abs_charge}).")
    else:
        parser.print_help()
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_bucket = defaultdict(list)
    n_reclassified = 0
    n_rejected_charge = 0
    for name, nx_smi, charge in results:
        # Classify by the ACTUAL RDKit net charge of the final SMILES, not by
        # the internally-computed sum: RDKit's sanitization/canonicalization
        # can reassign formal charges (e.g. on delocalized/aromatic nitrogen
        # rings), so the two can diverge. Trusting the computed sum here filed
        # a small fraction of structures under the wrong family, which then
        # got optimized with the wrong charge downstream. Re-read to be safe.
        m = Chem.MolFromSmiles(nx_smi)
        if m is None:
            continue
        real_charge = Chem.GetFormalCharge(m)
        if abs(real_charge) > args.max_abs_charge:
            n_rejected_charge += 1
            continue
        if real_charge != charge:
            n_reclassified += 1
        bucket = (n_size(nx_smi), charge_family(real_charge))
        by_bucket[bucket].append((name, nx_smi, real_charge))

    if n_reclassified or n_rejected_charge:
        print(f"[charge check] {n_reclassified} structure(s) reclassified to match "
              f"their true RDKit charge; {n_rejected_charge} rejected for "
              f"|charge| > {args.max_abs_charge}.")

    for (n, family), entries in sorted(by_bucket.items()):
        out_path = out_dir / f"N{n}_{family}_from_cxhx.smi"
        seen_smiles = set()
        unique_entries = []
        for name, nx_smi, charge in entries:
            if nx_smi in seen_smiles:
                continue
            seen_smiles.add(nx_smi)
            unique_entries.append((name, nx_smi, charge))
        with open(out_path, "a") as fh:
            for name, nx_smi, charge in unique_entries:
                fh.write(f"{nx_smi}\n")
        example_names = ", ".join(name for name, _, _ in unique_entries[:5])
        more = f" (+{len(unique_entries) - 5} more)" if len(unique_entries) > 5 else ""
        print(f"N{n} [{family}]: {len(unique_entries)} structure(s) -> {out_path} "
              f"({example_names}{more})")


if __name__ == "__main__":
    main()
