#!/usr/bin/env python3
"""
relax_imported_structures.py
=============================

Relaxation GFN2-xTB (via tblite, meme routine que polyN_pipeline.py) des
structures neutres importees dans structures_externes/{N_csp,polyN_study}/,
en vue de la comparaison a trois sources (biblio / N_csp / polyN_study)
demandee pour le rapport. Les structures biblio ont deja ete relaxees
separement (biblio_polyN/xtb_work/, via le CLI xtb) -- non retouchees ici.

Toutes les structures importees sont des clusters N_n neutres (aucune
variante chargee dans ces deux sources) : charge=0, uhf=1 si n impair
(nombre d'electrons impair, N=7 e- par atome), uhf=0 sinon.

Usage: /Users/akapulco/miniconda3/bin/python3 relax_imported_structures.py
"""

from __future__ import annotations

import csv
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from polyN_pipeline import _tblite_optimize  # noqa: E402

ROOT = Path(__file__).parent
SOURCES = ["N_csp", "polyN_study"]
WORKDIR = ROOT / "structures_externes" / "xtb_relax"
GFN = 2
OPT_LEVEL = "tight"
MAX_STEPS = 500


def load_manifest(source: str) -> list[dict]:
    path = ROOT / "structures_externes" / source / "manifest.csv"
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def uhf_for_n(n_atoms: int) -> int:
    return 1 if n_atoms % 2 else 0


def _worker(args):
    source, row, xyz_path = args
    n_atoms = int(row["n_atoms"])
    charge = 0
    uhf = uhf_for_n(n_atoms)
    tag = f"{source}_{Path(xyz_path).stem}"
    run_dir = WORKDIR / source / Path(xyz_path).stem
    t0 = time.time()
    ok, e_ha, xyz_out = _tblite_optimize(
        str(xyz_path), run_dir, charge, uhf,
        gfn=GFN, opt_level=OPT_LEVEL, max_steps=MAX_STEPS,
        out_name="xtbopt.xyz",
    )
    dt = time.time() - t0
    return {
        "source": source,
        "tag": tag,
        "xyz_file": str(xyz_path),
        "n_atoms": n_atoms,
        "topology_label": row.get("topology_label", ""),
        "charge": charge,
        "uhf": uhf,
        "converged": ok,
        "e_xtb_hartree": e_ha,
        "seconds": round(dt, 2),
    }


def main():
    tasks = []
    for source in SOURCES:
        rows = load_manifest(source)
        xyz_dir = ROOT / "structures_externes" / source / "xyz"
        for row in rows:
            xyz_rel = row["xyz_file"]
            xyz_path = ROOT / "structures_externes" / source / xyz_rel
            if not xyz_path.exists():
                continue
            tasks.append((source, row, xyz_path))

    print(f"{len(tasks)} structures a relaxer (GFN{GFN}-xTB, opt_level={OPT_LEVEL}).")

    # Reference N2 (charge 0, uhf 0) pour delta_E/atom -- coherent avec la
    # convention neutre du pipeline principal (relatif a N2).
    n2_dir = WORKDIR / "_ref_N2"
    n2_dir.mkdir(parents=True, exist_ok=True)
    n2_xyz = n2_dir / "N2.xyz"
    n2_xyz.write_text("2\nN2 reference\nN 0.0 0.0 0.0\nN 0.0 0.0 1.0977\n")
    ok_ref, e_n2_ha, _ = _tblite_optimize(
        str(n2_xyz), n2_dir, 0, 0, gfn=GFN, opt_level=OPT_LEVEL,
        max_steps=MAX_STEPS, out_name="xtbopt.xyz",
    )
    if not ok_ref:
        print("ERREUR: reference N2 n'a pas converge -- arret.")
        return
    e_n2_per_atom_ha = e_n2_ha / 2
    print(f"Reference N2 (GFN{GFN}-xTB): {e_n2_ha:.6f} Ha "
          f"({e_n2_per_atom_ha:.6f} Ha/atom)")

    n_jobs = max(1, (mp.cpu_count() or 2) - 1)
    print(f"Lancement sur {n_jobs} coeurs en parallele...")

    results = []
    t_start = time.time()
    with mp.Pool(n_jobs) as pool:
        for i, res in enumerate(pool.imap_unordered(_worker, tasks), 1):
            results.append(res)
            if i % 25 == 0 or i == len(tasks):
                elapsed = time.time() - t_start
                n_ok = sum(1 for r in results if r["converged"])
                print(f"  [{i}/{len(tasks)}] convergees: {n_ok} "
                      f"(elapsed {elapsed:.0f}s)")

    HARTREE_TO_EV = 27.211386245988
    out_path = ROOT / "structures_externes" / "xtb_relax_results.csv"
    with open(out_path, "w", newline="") as fh:
        fieldnames = ["source", "tag", "xyz_file", "n_atoms", "topology_label",
                      "charge", "uhf", "converged", "e_xtb_hartree",
                      "e_xtb_eV_per_atom", "delta_E_eV_per_atom_vs_N2", "seconds"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            if r["converged"] and r["e_xtb_hartree"] is not None:
                e_per_atom_ha = r["e_xtb_hartree"] / r["n_atoms"]
                r["e_xtb_eV_per_atom"] = round(e_per_atom_ha * HARTREE_TO_EV, 5)
                r["delta_E_eV_per_atom_vs_N2"] = round(
                    (e_per_atom_ha - e_n2_per_atom_ha) * HARTREE_TO_EV, 5)
            else:
                r["e_xtb_eV_per_atom"] = ""
                r["delta_E_eV_per_atom_vs_N2"] = ""
            writer.writerow(r)

    n_ok = sum(1 for r in results if r["converged"])
    total_time = time.time() - t_start
    print(f"\nTermine en {total_time:.0f}s : {n_ok}/{len(tasks)} structures "
          f"relaxees avec succes.")
    print(f"Resultats: {out_path}")


if __name__ == "__main__":
    main()
