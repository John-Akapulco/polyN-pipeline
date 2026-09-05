#!/usr/bin/env python3
"""
relax_exact_structures.py
===========================

Les 45 structures biblio "Niveau 1" (geometrie exacte reconstruite depuis
les parametres publies) n'ont jamais ete passees par GFN2-xTB -- seules
les 25 "Niveau 2" (topologie seule) l'ont ete, puisqu'elles en avaient
besoin pour obtenir une geometrie. Pour un rapport comparant des energies
relatives au sein d'une meme famille (formule, charge), il faut un niveau
de theorie commun a TOUTES les structures : ce script relaxe donc aussi
les 45 structures exactes en GFN2-xTB (tblite) et verifie leurs
frequences, exactement comme pour les 25 autres et pour l'archive
complementaire.
"""

from __future__ import annotations

import csv
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ase.io import read
from frequency_check import verify_and_relax_to_minimum  # noqa: E402
from polyN_pipeline import _tblite_optimize  # noqa: E402

ROOT = Path(__file__).parent.parent
XYZDIR = ROOT / "xyz"
WORKDIR = ROOT / "exact_relax_work"
GFN = 2


def _worker(entry):
    eid, charge, mult = entry["id"], entry["charge"], entry["mult"]
    uhf = mult - 1
    xyz_path = XYZDIR / f"{eid}.xyz"
    if not xyz_path.exists():
        return {**entry, "status": "missing_xyz"}

    run_dir = WORKDIR / eid
    ok, e_ha, xyz_out = _tblite_optimize(
        str(xyz_path), run_dir, charge, uhf,
        gfn=GFN, opt_level="tight", max_steps=500, out_name="xtbopt.xyz",
    )
    if not ok:
        return {**entry, "status": "opt_failed"}

    atoms = read(xyz_out)
    freq_dir = WORKDIR / eid / "freq"
    try:
        result = verify_and_relax_to_minimum(atoms, charge=charge, uhf=uhf, gfn=GFN,
                                              work_dir=freq_dir, max_hops=5)
    except Exception as exc:
        return {**entry, "status": f"freq_error: {exc}"}

    n_atoms = len(atoms)
    return {
        **entry,
        "status": "ok",
        "is_true_minimum": result["is_minimum"],
        "hops_needed": result["hops_needed"],
        "e_xtb_hartree": result["final_energy_ev"] / 27.211386245988,
        "e_xtb_eV_per_atom": round(result["final_energy_ev"] / n_atoms, 6),
    }


def main():
    manifest = json.load(open(ROOT / "build" / "manifest.json"))
    exact = [e for e in manifest if e["status"] == "exact"]
    print(f"{len(exact)} structures 'exactes' a relaxer en GFN{GFN}-xTB.")

    n_jobs = max(1, (mp.cpu_count() or 2) - 1)
    results = []
    t0 = time.time()
    with mp.Pool(n_jobs) as pool:
        for i, res in enumerate(pool.imap_unordered(_worker, exact), 1):
            results.append(res)
            if i % 10 == 0 or i == len(exact):
                print(f"  [{i}/{len(exact)}] elapsed {time.time()-t0:.0f}s")

    out_path = ROOT / "exact_structures_xtb_results.csv"
    fieldnames = ["id", "formula", "charge", "mult", "pg", "method", "source", "n",
                  "status", "is_true_minimum", "hops_needed", "e_xtb_hartree", "e_xtb_eV_per_atom"]
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    n_ok = sum(1 for r in results if r["status"] == "ok")
    print(f"Termine en {time.time()-t0:.0f}s : {n_ok}/{len(exact)} reussis.")
    print(f"Resultats: {out_path}")


if __name__ == "__main__":
    main()
