#!/usr/bin/env python3
"""
check_xyz_purity.py
=====================

Scan .xyz geometry files and report the molecular formula of each,
flagging any file that contains an element OTHER than the expected one
(default: N). Use this to catch contamination -- e.g. stray hydrogens
introduced when a bare (non-bracket) "N" SMILES atom with fewer than 3
explicit bonds gets an implicit hydrogen silently added by RDKit's
valence model, which then survives all the way through to the xtb
calculation as a real atom of a DIFFERENT chemical species than intended.

Usage
-----
    # Scan every .xyz file found recursively under a directory:
    python check_xyz_purity.py --dir ./resultats/production_2026-07-03

    # Scan exactly the files referenced in a results.csv (recommended,
    # since that's the authoritative list of what the pipeline reported on):
    python check_xyz_purity.py --results-csv ./resultats/production_2026-07-03/results.csv

    # Restrict to a specific expected element (default: N)
    python check_xyz_purity.py --dir . --element N
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    pd = None


def read_xyz_formula(xyz_path: Path) -> Counter:
    """Return element counts from an .xyz file without any chemistry
    toolkit dependency (just parses the plain-text format directly)."""
    with open(xyz_path) as fh:
        lines = fh.readlines()
    n_atoms = int(lines[0].strip())
    counts = Counter()
    for line in lines[2:2 + n_atoms]:
        parts = line.split()
        if not parts:
            continue
        counts[parts[0]] += 1
    return counts


def formula_string(counts: Counter) -> str:
    return "".join(f"{el}{n}" for el, n in sorted(counts.items()))


def check_file(xyz_path: Path, expected_element: str) -> dict:
    try:
        counts = read_xyz_formula(xyz_path)
    except Exception as exc:
        return dict(path=str(xyz_path), formula="?", contaminated=None, error=str(exc))
    other_elements = {el: n for el, n in counts.items() if el != expected_element}
    return dict(
        path=str(xyz_path),
        formula=formula_string(counts),
        n_target=counts.get(expected_element, 0),
        contaminated=bool(other_elements),
        other_elements=other_elements,
        error=None,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=str, default=None,
                         help="Recursively scan every .xyz file under this directory")
    parser.add_argument("--results-csv", type=str, default=None,
                         help="Scan exactly the files listed in this results.csv's 'xyz' column")
    parser.add_argument("--element", type=str, default="N",
                         help="Expected (only allowed) element symbol (default: N)")
    parser.add_argument("--quiet", action="store_true",
                         help="Only print contaminated files, not the full clean list")
    args = parser.parse_args()

    if not args.dir and not args.results_csv:
        raise SystemExit("Provide either --dir or --results-csv.")

    xyz_paths = []
    if args.dir:
        xyz_paths.extend(sorted(Path(args.dir).rglob("*.xyz")))
    if args.results_csv:
        if pd is None:
            raise SystemExit("pandas is required for --results-csv (pip install pandas)")
        df = pd.read_csv(args.results_csv)
        if "xyz" not in df.columns:
            raise SystemExit("No 'xyz' column found in the given results.csv.")
        xyz_paths.extend(Path(p) for p in df["xyz"].dropna().unique())

    if not xyz_paths:
        print("No .xyz files found.")
        return

    n_ok, n_bad, n_error = 0, 0, 0
    bad_rows = []
    for p in xyz_paths:
        result = check_file(p, args.element)
        if result["error"]:
            n_error += 1
            print(f"[ERROR]       {p}: {result['error']}")
            continue
        if result["contaminated"]:
            n_bad += 1
            bad_rows.append(result)
            print(f"[CONTAMINATED] {result['formula']:15s} {p}  "
                  f"-- unexpected element(s): {result['other_elements']}")
        else:
            n_ok += 1
            if not args.quiet:
                print(f"[ok]          {result['formula']:15s} {p}")

    print(f"\n{'='*60}")
    print(f"Total scanned: {len(xyz_paths)}   OK: {n_ok}   "
          f"CONTAMINATED: {n_bad}   errors: {n_error}")
    if bad_rows:
        print("\nFormulas involved in contamination:")
        for formula, count in Counter(r["formula"] for r in bad_rows).most_common():
            print(f"  {formula:15s} x{count}")


if __name__ == "__main__":
    main()
