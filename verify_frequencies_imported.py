#!/usr/bin/env python3
"""
verify_frequencies_imported.py
================================

Verification de frequences (Hessienne GFN2-xTB numerique) sur les 312
structures importees (N_csp, polyN_study) ayant converge lors de la
relaxation (cf. relax_imported_structures.py / structures_externes/
xtb_relax_results.csv). Reutilise verify_and_relax_to_minimum() de
frequency_check.py : si des modes imaginaires significatifs sont trouves,
les suit et reoptimise (jusqu'a max_hops), pour confirmer qu'on tient un
vrai minimum local et non un point selle non detecte par le gradient seul.

Usage: /Users/akapulco/miniconda3/bin/python3 verify_frequencies_imported.py
"""

from __future__ import annotations

import csv
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ase.io import read
from frequency_check import verify_and_relax_to_minimum  # noqa: E402

ROOT = Path(__file__).parent
RELAX_WORKDIR = ROOT / "structures_externes" / "xtb_relax"
FREQ_WORKDIR = ROOT / "structures_externes" / "freq_verify"
GFN = 2
MAX_HOPS = 5

HARTREE_TO_EV = 27.211386245988


def _worker(row: dict):
    source = row["source"]
    orig_stem = Path(row["xyz_file"]).stem
    xtbopt_xyz = RELAX_WORKDIR / source / orig_stem / "xtbopt.xyz"
    if not xtbopt_xyz.exists():
        return {**row, "freq_status": "missing_xtbopt_xyz"}

    atoms = read(str(xtbopt_xyz))
    charge = int(row["charge"])
    uhf = int(row["uhf"])
    work_dir = FREQ_WORKDIR / source / orig_stem

    t0 = time.time()
    try:
        result = verify_and_relax_to_minimum(
            atoms, charge=charge, uhf=uhf, gfn=GFN,
            work_dir=work_dir, max_hops=MAX_HOPS,
        )
    except Exception as exc:
        return {**row, "freq_status": f"error: {exc}"}
    dt = time.time() - t0

    n_imag_hop0 = result["history"][0]["n_imaginary"]
    return {
        **row,
        "freq_status": "ok",
        "is_true_minimum": result["is_minimum"],
        "hops_needed": result["hops_needed"],
        "n_imaginary_before_hopping": n_imag_hop0,
        "e_final_eV_per_atom": round(
            result["final_energy_ev"] / int(row["n_atoms"]), 6),
        "e_shift_eV_per_atom": round(
            (result["final_energy_ev"] / int(row["n_atoms"])) -
            float(row["e_xtb_eV_per_atom"]), 6),
        "freq_seconds": round(dt, 1),
    }


def main():
    in_path = ROOT / "structures_externes" / "xtb_relax_results.csv"
    with open(in_path, newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["converged"] == "True"]
    print(f"{len(rows)} structures convergees a verifier (frequences GFN{GFN}-xTB).")

    n_jobs = max(1, (mp.cpu_count() or 2) - 1)
    print(f"Lancement sur {n_jobs} coeurs en parallele...")

    results = []
    t_start = time.time()
    with mp.Pool(n_jobs) as pool:
        for i, res in enumerate(pool.imap_unordered(_worker, rows), 1):
            results.append(res)
            if i % 20 == 0 or i == len(rows):
                elapsed = time.time() - t_start
                n_min = sum(1 for r in results if r.get("is_true_minimum") is True)
                n_hopped = sum(1 for r in results if r.get("hops_needed", 0) and r["hops_needed"] > 0)
                print(f"  [{i}/{len(rows)}] vrais minima: {n_min}, "
                      f"ayant necessite >=1 hop: {n_hopped} (elapsed {elapsed:.0f}s)")

    out_path = ROOT / "structures_externes" / "freq_verification_results.csv"
    fieldnames = ["source", "tag", "xyz_file", "n_atoms", "topology_label", "charge",
                  "uhf", "e_xtb_eV_per_atom", "freq_status", "is_true_minimum",
                  "hops_needed", "n_imaginary_before_hopping", "e_final_eV_per_atom",
                  "e_shift_eV_per_atom", "freq_seconds"]
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    total_time = time.time() - t_start
    n_ok = sum(1 for r in results if r.get("freq_status") == "ok")
    n_true_min = sum(1 for r in results if r.get("is_true_minimum") is True)
    n_hopped = sum(1 for r in results if r.get("hops_needed", 0) and r["hops_needed"] > 0)
    n_unresolved = sum(1 for r in results if r.get("freq_status") == "ok" and r.get("is_true_minimum") is False)
    n_errors = sum(1 for r in results if r.get("freq_status") != "ok")
    print(f"\nTermine en {total_time:.0f}s.")
    print(f"  Calculs reussis      : {n_ok}/{len(rows)}")
    print(f"  Vrais minima (0 hop) : {sum(1 for r in results if r.get('is_true_minimum') is True and r.get('hops_needed')==0)}")
    print(f"  Vrais minima (>=1 hop, structure corrigee) : {n_hopped - sum(1 for r in results if r.get('is_true_minimum') is False and r.get('hops_needed',0)>0)}")
    print(f"  Points selles non resolus (max_hops atteint) : {n_unresolved}")
    print(f"  Erreurs / xyz manquant : {n_errors}")
    print(f"Resultats: {out_path}")


if __name__ == "__main__":
    main()
