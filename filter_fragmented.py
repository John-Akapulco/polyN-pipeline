#!/usr/bin/env python3
"""
filter_fragmented.py
====================

Remove fragmented structures from an existing pipeline output, without
re-running the whole pipeline. A "fragmented" structure is one whose atoms
split into more than one connected component under a distance-based bond
criterion -- the canonical case being a detached N2 in van der Waals contact
with the rest (e.g. an "N8" that is really N6 + N2). These are not genuine
single-molecule Nx allotropes and should not enter a training set.

What it does:
  - reads results.csv, checks each structure's .xyz for fragmentation,
  - writes a cleaned results.csv (default: results_clean.csv) with the
    fragmented rows removed and ranks recomputed per (formula, family),
  - optionally rebuilds a clean best_structures/ directory,
  - prints a report of what was removed.

Usage
-----
    python3 filter_fragmented.py --results resultats/seeds_pubchem/results.csv
    python3 filter_fragmented.py --results .../results.csv --threshold 2.0 \\
            --rebuild-best-dir
    python3 filter_fragmented.py --results .../results.csv --in-place

Bond threshold guidance: a real N-N bond is <= ~1.6 A; a van der Waals
N...N contact is ~3 A. The default 2.0 A cleanly separates the two.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def read_xyz_coords(path: str):
    """Return an (n,3) array of coordinates from an .xyz (extended-xyz safe)."""
    lines = Path(path).read_text().splitlines()
    n = int(lines[0].split()[0])
    coords = []
    for ln in lines[2:2 + n]:
        p = ln.split()
        coords.append([float(p[1]), float(p[2]), float(p[3])])
    return np.array(coords)


def fragment_sizes(coords: np.ndarray, threshold: float):
    """Return the sorted list of connected-component sizes."""
    n = len(coords)
    d = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if d[i, j] < threshold:
                parent[find(i)] = find(j)
    comps = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)
    return sorted(len(v) for v in comps.values())


def main():
    ap = argparse.ArgumentParser(
        description="Remove fragmented (multi-component) structures from a "
                    "pipeline results.csv.")
    ap.add_argument("--results", required=True, help="Path to results.csv")
    ap.add_argument("--threshold", type=float, default=2.0,
                    help="N-N bond distance cutoff in angstrom (default 2.0). "
                         "Atoms closer than this are 'bonded'; a structure with "
                         ">1 resulting component is fragmented.")
    ap.add_argument("--out", default=None,
                    help="Output cleaned CSV (default: results_clean.csv "
                         "next to the input).")
    ap.add_argument("--in-place", action="store_true",
                    help="Overwrite the input results.csv (a .bak backup is "
                         "made first).")
    ap.add_argument("--rebuild-best-dir", action="store_true",
                    help="Also rebuild a clean best_structures/ directory "
                         "(named best_structures_clean/) with only intact "
                         "structures, re-ranked.")
    args = ap.parse_args()

    results_path = Path(args.results)
    df = pd.read_csv(results_path)
    n0 = len(df)

    intact_mask = []
    removed = []
    for _, row in df.iterrows():
        xyz = row.get("xyz")
        try:
            coords = read_xyz_coords(str(xyz))
            sizes = fragment_sizes(coords, args.threshold)
        except Exception:
            sizes = [1]  # unreadable -> keep (don't reject on I/O error)
        is_intact = len(sizes) == 1
        intact_mask.append(is_intact)
        if not is_intact:
            removed.append((row.get("formula"), row.get("family"),
                            row.get("rank"), sizes,
                            row.get("e_reaction_kcalmol")))

    df_clean = df[pd.Series(intact_mask, index=df.index)].copy()

    # Recompute ranks per (formula, family) after removal.
    if "e_xtb_hartree" in df_clean.columns:
        df_clean["rank"] = (df_clean.groupby(["formula", "family"])["e_xtb_hartree"]
                            .rank(method="first").astype(int))

    # Report.
    print(f"Read {n0} structures from {results_path}")
    print(f"Fragmentation cutoff: {args.threshold} A")
    print(f"Removed {len(removed)} fragmented structure(s); "
          f"{len(df_clean)} intact remaining.\n")
    if removed:
        print("Removed structures (fragment sizes):")
        for formula, family, rank, sizes, e in removed:
            e_txt = f"{e:+.1f}" if pd.notna(e) else "n/a"
            print(f"  {formula}_{family} #{rank}: {sizes}  (E={e_txt} kcal/mol)")
        print()

    # Write cleaned CSV.
    if args.in_place:
        backup = results_path.with_suffix(results_path.suffix + ".bak")
        shutil.copyfile(results_path, backup)
        df_clean.to_csv(results_path, index=False)
        print(f"Overwrote {results_path} (backup at {backup})")
        out_csv = results_path
    else:
        out_csv = Path(args.out) if args.out else \
            results_path.with_name(results_path.stem + "_clean.csv")
        df_clean.to_csv(out_csv, index=False)
        print(f"Cleaned CSV -> {out_csv}")

    # Optionally rebuild a clean best_structures directory.
    if args.rebuild_best_dir:
        best_dir = results_path.parent / "best_structures_clean"
        best_dir.mkdir(parents=True, exist_ok=True)
        n_exported = 0
        for _, row in df_clean.iterrows():
            xyz = row.get("xyz")
            if not xyz or not Path(xyz).exists():
                continue
            dest = best_dir / f"{row['formula']}_{int(row['rank']):03d}.xyz"
            shutil.copyfile(xyz, dest)
            n_exported += 1
        print(f"Rebuilt {best_dir} with {n_exported} intact structures.")


if __name__ == "__main__":
    main()
