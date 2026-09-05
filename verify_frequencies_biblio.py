#!/usr/bin/env python3
"""
verify_frequencies_biblio.py
==============================

Meme verification de frequences (Hessienne GFN2-xTB, suivi de mode
imaginaire si necessaire) que verify_frequencies_imported.py, appliquee
cette fois aux 25 structures extraites des deux articles de reference
(biblio_polyN/xtb_work/), deja relaxees via le binaire xtb en ligne de
commande. Charge et multiplicite reprises telles que declarees dans
biblio_polyN/build/molecules.py (pas devinees par parite -- certaines
structures, ex. N4_C2v_butterfly, sont des triplets non evidents a deviner).

Usage: /Users/akapulco/miniconda3/bin/python3 verify_frequencies_biblio.py
"""

from __future__ import annotations

import csv
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ase import Atoms
from frequency_check import verify_and_relax_to_minimum  # noqa: E402

ROOT = Path(__file__).parent
XTBWORK = ROOT / "biblio_polyN" / "xtb_work"
FREQ_WORKDIR = ROOT / "biblio_polyN" / "freq_verify"
GFN = 2
MAX_HOPS = 5

# (name, charge, mult) -- repris de biblio_polyN/build/molecules.py
ENTRIES = [
    ("N10_C3_cap", 0, 1),
    ("N10_cage_C2v", 0, 1),
    ("N10_Cs_ring_chain", 0, 1),
    ("N10-_D2h_perp_rings", -1, 2),
    ("N10+_rings", 1, 2),
    ("N12_C2h_dipentazolyldiazene", 0, 1),
    ("N4_C2v_butterfly", 0, 3),
    ("N6_C2_book", 0, 1),
    ("N6_D2_twisted", 0, 1),
    ("N6-_C2_linked_triangles", -1, 2),
    ("N6+_C2h_nonplanar", 1, 2),
    ("N7_C2v_logcarrier", 0, 2),
    ("N7_Cs_ring_chain", 0, 2),
    ("N8_C2h_ladder", 0, 1),
    ("N8_C2v_branched", 0, 1),
    ("N8_C2v_ring", 0, 1),
    ("N8_Cs_azidopentazole", 0, 1),
    ("N8_Cs_pentagonal", 0, 1),
    ("N8_Cs_ZEE", 0, 1),
    ("N8_D2d_octaazacyclooctatetraene", 0, 1),
    ("N8_D2h_pentalene", 0, 1),
    ("N8_D2h_ring_pendant", 0, 1),
    ("N9_fused_rings", 0, 2),
    ("N9-_ring_chain", -1, 1),
    ("N9+_fused_rings", 1, 1),
]


def read_last_frame(log_path: Path) -> Atoms:
    lines = log_path.read_text().splitlines()
    n = int(lines[0].strip())
    block = len(lines) // (n + 2) if len(lines) % (n + 2) == 0 else None
    # dernier bloc de n+2 lignes (le format xtbopt.log concatene des
    # frames identiques au format xyz standard)
    last_block = lines[-(n + 2):]
    symbols, positions = [], []
    for line in last_block[2:2 + n]:
        p = line.split()
        symbols.append(p[0])
        positions.append([float(p[1]), float(p[2]), float(p[3])])
    return Atoms(symbols=symbols, positions=positions)


def _worker(entry):
    name, charge, mult = entry
    uhf = mult - 1
    log_path = XTBWORK / name / "xtbopt.log"
    atoms = read_last_frame(log_path)
    work_dir = FREQ_WORKDIR / name

    t0 = time.time()
    try:
        result = verify_and_relax_to_minimum(
            atoms, charge=charge, uhf=uhf, gfn=GFN,
            work_dir=work_dir, max_hops=MAX_HOPS,
        )
    except Exception as exc:
        return {"name": name, "charge": charge, "mult": mult, "n_atoms": len(atoms),
                "freq_status": f"error: {exc}"}
    dt = time.time() - t0

    n_atoms = len(atoms)
    e_initial_ev_per_atom = atoms.get_potential_energy() if False else None
    return {
        "name": name,
        "charge": charge,
        "mult": mult,
        "n_atoms": n_atoms,
        "freq_status": "ok",
        "is_true_minimum": result["is_minimum"],
        "hops_needed": result["hops_needed"],
        "n_imaginary_before_hopping": result["history"][0]["n_imaginary"],
        "e_final_eV_per_atom": round(result["final_energy_ev"] / n_atoms, 6),
        "freq_seconds": round(dt, 1),
    }


def main():
    print(f"{len(ENTRIES)} structures biblio a verifier (frequences GFN{GFN}-xTB).")
    n_jobs = max(1, (mp.cpu_count() or 2) - 1)

    results = []
    t_start = time.time()
    with mp.Pool(min(n_jobs, len(ENTRIES))) as pool:
        for res in pool.imap_unordered(_worker, ENTRIES):
            results.append(res)
            print(f"  {res['name']}: status={res['freq_status']}, "
                  f"minimum={res.get('is_true_minimum')}, hops={res.get('hops_needed')}")

    out_path = ROOT / "biblio_polyN" / "freq_verification_results_biblio.csv"
    fieldnames = ["name", "charge", "mult", "n_atoms", "freq_status",
                  "is_true_minimum", "hops_needed", "n_imaginary_before_hopping",
                  "e_final_eV_per_atom", "freq_seconds"]
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    total_time = time.time() - t_start
    n_ok = sum(1 for r in results if r.get("freq_status") == "ok")
    n_direct = sum(1 for r in results if r.get("is_true_minimum") is True and r.get("hops_needed") == 0)
    n_hopped = sum(1 for r in results if r.get("is_true_minimum") is True and r.get("hops_needed", 0) > 0)
    n_unresolved = sum(1 for r in results if r.get("freq_status") == "ok" and r.get("is_true_minimum") is False)
    n_errors = len(results) - n_ok
    print(f"\nTermine en {total_time:.0f}s.")
    print(f"  Calculs reussis: {n_ok}/{len(ENTRIES)}")
    print(f"  Minima directs (0 hop): {n_direct}")
    print(f"  Minima corriges (>=1 hop): {n_hopped}")
    print(f"  Points-selles non resolus: {n_unresolved}")
    print(f"  Erreurs: {n_errors}")
    print(f"Resultats: {out_path}")


if __name__ == "__main__":
    main()
