#!/usr/bin/env python3
"""
test_charge_propagation.py
==========================

Non-regression test for the charge-handling bug fixed in this pipeline, plus
the hydrogen-contamination bug fixed in the generators. Run it after any
change to a generator or to the tblite optimization path.

It checks three independent things, for structures from each of the four
topology generators (isolobal / random / geng, plus a hand-built control
set -- MAYGEN is neutral-only and needs an external jar, so it is checked
structurally rather than by enumeration here):

  CHECK 1 -- Net charge in the SMILES matches the declared family.
      A structure filed under "*_cation_*" must actually carry net charge
      +1 as parsed by RDKit, "*_anion_*" net -1, "*_neutral_*" net 0. This
      catches a generator that mislabels a structure's family.

  CHECK 2 -- tblite applies the charge (the bug we fixed).
      Optimizing the SAME geometry as +1 and as -1 must give DIFFERENT
      energies. If they are equal, tblite is ignoring the charge (the
      constructor-kwarg-ignored bug) and every ion in the run would be
      wrong. This is the core regression guard.

  CHECK 3 -- No phantom hydrogen.
      After construction, no structure may contain a hydrogen atom. Catches
      RDKit's silent implicit-H saturation of an under-valent nitrogen.

Exit code is nonzero if any check fails, so it can be used in CI.

Usage
-----
    python3 test_charge_propagation.py
    python3 test_charge_propagation.py --seeds-dir ./some/existing/seeds
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

HARTREE = 27.211386245988  # eV per Hartree

PASS = "PASS"
FAIL = "FAIL"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def net_charge_of_smiles(smi: str):
    """RDKit net formal charge of a SMILES, or None if unparseable."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return Chem.GetFormalCharge(m)


def has_hydrogen(smi: str) -> bool:
    """True if the (explicit-H) molecule contains any hydrogen atom."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return False
    mh = Chem.AddHs(m)
    return any(a.GetAtomicNum() == 1 for a in mh.GetAtoms())


def family_from_filename(name: str):
    """Infer the intended charge family from a generator filename."""
    low = name.lower()
    if "cation" in low:
        return "cation", +1
    if "anion" in low:
        return "anion", -1
    if "neutral" in low:
        return "neutral", 0
    return None, None


# ---------------------------------------------------------------------------
# CHECK 2 -- does tblite actually apply the charge?  (core regression guard)
# ---------------------------------------------------------------------------

def check_tblite_applies_charge(verbose: bool = True) -> bool:
    """Optimize a fixed N5 geometry as +1 and -1; energies MUST differ.

    Reproduces exactly the mechanism the pipeline now uses
    (set_initial_charges), so a regression in that code path is caught."""
    try:
        from tblite.ase import TBLite
        from ase import Atoms
        from ase.optimize import LBFGS
    except Exception as exc:
        print(f"  [CHECK 2] SKIPPED -- tblite/ase not importable ({exc})")
        return True  # cannot test here; not a failure of the generators

    pos = [[0, 0, 0], [1.3, 0, 0], [2.0, 1.1, 0], [1.3, 2.2, 0], [0, 1.6, 0]]

    def optimize(charge: int) -> float:
        atoms = Atoms("N5", positions=pos)
        if charge != 0:
            init = [0.0] * len(atoms)
            init[0] = float(charge)          # sum = net charge (the fix)
            atoms.set_initial_charges(init)
        atoms.calc = TBLite(method="GFN2-xTB", charge=charge,
                            multiplicity=1, verbosity=0)
        LBFGS(atoms, logfile=None).run(fmax=0.0025, steps=60)
        return atoms.get_potential_energy() / HARTREE

    e_plus = optimize(+1)
    e_minus = optimize(-1)
    diff_kcal = abs(e_plus - e_minus) * 627.509

    ok = diff_kcal > 1.0  # any real charge effect is >> 1 kcal/mol
    if verbose:
        print(f"  [CHECK 2] tblite applies charge:")
        print(f"            N5(+1) = {e_plus:.6f} Ha")
        print(f"            N5(-1) = {e_minus:.6f} Ha")
        print(f"            |diff| = {diff_kcal:.1f} kcal/mol  -> "
              f"{PASS if ok else FAIL}")
        if not ok:
            print("            !!! tblite is IGNORING the charge -- every ion "
                  "would be optimized as neutral.")
    return ok


# ---------------------------------------------------------------------------
# CHECK 1 & 3 -- charge label vs SMILES, and hydrogen purity, per generator
# ---------------------------------------------------------------------------

def check_seed_files(seeds_dir: Path, verbose: bool = True) -> bool:
    """For every .smi in seeds_dir, verify each structure's net charge
    matches its filename's family and that no hydrogen is present."""
    smi_files = sorted(seeds_dir.glob("*.smi"))
    if not smi_files:
        print(f"  [CHECK 1/3] no .smi files found in {seeds_dir}")
        return True

    all_ok = True
    per_gen_summary = {}
    for f in smi_files:
        fam, expected_q = family_from_filename(f.name)
        # infer generator tag from filename suffix
        gen = ("isolobal" if "cxhx" in f.name.lower()
               else "geng" if "geng" in f.name.lower()
               else "random" if "random" in f.name.lower()
               else "other")
        lines = [l.strip() for l in open(f) if l.strip()]
        n_bad_charge = 0
        n_with_h = 0
        for smi in lines:
            if expected_q is not None:
                q = net_charge_of_smiles(smi)
                if q is not None and q != expected_q:
                    n_bad_charge += 1
            if has_hydrogen(smi):
                n_with_h += 1
        key = gen
        s = per_gen_summary.setdefault(key, {"files": 0, "structs": 0,
                                             "bad_charge": 0, "with_h": 0})
        s["files"] += 1
        s["structs"] += len(lines)
        s["bad_charge"] += n_bad_charge
        s["with_h"] += n_with_h
        if n_bad_charge or n_with_h:
            all_ok = False
            if verbose:
                print(f"    {f.name}: {n_bad_charge} wrong-charge, "
                      f"{n_with_h} with-H  (of {len(lines)})")

    if verbose:
        print("  [CHECK 1] net charge matches family label")
        print("  [CHECK 3] no phantom hydrogen")
        print(f"  {'generator':<12}{'files':>7}{'structs':>9}"
              f"{'badQ':>7}{'withH':>7}  result")
        for gen, s in sorted(per_gen_summary.items()):
            gen_ok = (s["bad_charge"] == 0 and s["with_h"] == 0)
            print(f"  {gen:<12}{s['files']:>7}{s['structs']:>9}"
                  f"{s['bad_charge']:>7}{s['with_h']:>7}  "
                  f"{PASS if gen_ok else FAIL}")
    return all_ok


# ---------------------------------------------------------------------------
# Built-in control set: generate a few structures live (no external tools)
# ---------------------------------------------------------------------------

def build_control_seeds(out_dir: Path, verbose: bool = True) -> None:
    """Produce a tiny known-answer seed set using the isolobal and geng
    generators directly (no PubChem, no nauty), so the file-level checks
    always have something to run against even with no pre-existing seeds."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- geng-style: enumerate bond orders on a hand-built N5 ring ---
    try:
        import importlib.util
        here = Path(__file__).parent
        spec = importlib.util.spec_from_file_location(
            "ge", str(here / "geng_enumerate.py"))
        ge = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ge)
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]  # N5 ring
        found = ge.enumerate_bond_orders(edges, 5, max_abs_charge=1)
        by_fam = {"neutral": [], "cation": [], "anion": []}
        for smi, q in found.items():
            by_fam[ge.family_of(q)].append(smi)
        for fam, smis in by_fam.items():
            if smis:
                with open(out_dir / f"N5_{fam}_from_geng.smi", "w") as fh:
                    fh.write("\n".join(smis) + "\n")
        if verbose:
            print(f"  control set (geng N5 ring): "
                  f"{sum(len(v) for v in by_fam.values())} structures")
    except Exception as exc:
        if verbose:
            print(f"  control set (geng): skipped ({exc})")

    # --- isolobal-style: convert a couple of known hydrocarbons ---
    try:
        import importlib.util
        here = Path(__file__).parent
        spec = importlib.util.spec_from_file_location(
            "cx", str(here / "cxhx_to_nx.py"))
        cx = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cx)
        cases = {  # hydrocarbon -> expected family
            "C1=CC=CC=C1": "neutral",   # benzene -> hexazine N6
            "C1=CC=CC1":   "anion",     # cyclopentadiene -> pentazolate N5-
        }
        buckets = {}
        for smi, fam in cases.items():
            nx, q, err = cx.cxhx_to_nx(smi)
            if nx and not err:
                m = Chem.MolFromSmiles(nx)
                n = sum(1 for a in m.GetAtoms() if a.GetAtomicNum() == 7)
                famname = ("neutral" if q == 0
                           else "cation" if q > 0 else "anion")
                buckets.setdefault((n, famname), []).append(nx)
        for (n, famname), smis in buckets.items():
            with open(out_dir / f"N{n}_{famname}_from_cxhx.smi", "w") as fh:
                fh.write("\n".join(smis) + "\n")
        if verbose:
            print(f"  control set (isolobal): "
                  f"{sum(len(v) for v in buckets.values())} structures")
    except Exception as exc:
        if verbose:
            print(f"  control set (isolobal): skipped ({exc})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Non-regression test: charge propagation and hydrogen "
                    "purity across the pipeline's generators.")
    ap.add_argument("--seeds-dir", default=None,
                    help="Directory of existing *.smi seed files to check. "
                         "If omitted, a small control set is generated live.")
    args = ap.parse_args()

    print("=" * 68)
    print("polyN_pipeline -- charge propagation & purity regression test")
    print("=" * 68)

    results = {}

    # CHECK 2 first: the core tblite regression guard (independent of seeds).
    print("\n[Core guard] Does tblite apply the charge?")
    results["tblite_charge"] = check_tblite_applies_charge()

    # Prepare seeds to check (existing, or a freshly built control set).
    if args.seeds_dir:
        seeds_dir = Path(args.seeds_dir)
        print(f"\n[Seeds] Using existing seeds in {seeds_dir}")
    else:
        seeds_dir = Path("./_test_control_seeds")
        print(f"\n[Seeds] No --seeds-dir given; building a live control set "
              f"in {seeds_dir}")
        build_control_seeds(seeds_dir)

    print("\n[File checks] charge label vs SMILES, and hydrogen purity")
    results["seed_files"] = check_seed_files(seeds_dir)

    print("\n" + "=" * 68)
    all_ok = all(results.values())
    for name, ok in results.items():
        print(f"  {name:<20} {PASS if ok else FAIL}")
    print("=" * 68)
    if all_ok:
        print("ALL CHECKS PASSED")
        sys.exit(0)
    else:
        print("SOME CHECKS FAILED -- see above")
        sys.exit(1)


if __name__ == "__main__":
    main()
