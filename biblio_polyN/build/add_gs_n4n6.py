#!/usr/bin/env python3
"""
add_gs_n4n6.py
===============

Ajoute au corpus les 6 isomeres N4/N6 rapportes par [GS] (Glukhovtsev,
Jiao, von Rague Schleyer, Inorg. Chem. 1996, 35, 7124) Table 1/3 et
Figures 1/2 -- domaine manque lors de l'extraction initiale (seules les
donnees N8/N10/N12/N20 de cet article avaient ete extraites).

Geometries construites a partir des longueurs de liaison/angles publies
(Figure 1 pour N4 : structures 7 C2h triplet, 8 Td, 8a D2h ; Figure 2
pour N6 : structures 9 D6h, 10 D2, 10a C2h), valeurs Becke3LYP/6-311+G*
("en gras" sur les figures). Relaxees en GFN2-xTB puis verifiees en
frequence, comme le reste du corpus.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import geom
import numpy as np
from ase import Atoms
from frequency_check import verify_and_relax_to_minimum  # noqa: E402
from polyN_pipeline import _tblite_optimize  # noqa: E402

ROOT = Path(__file__).parent.parent
WORKDIR = ROOT / "gs_n4n6_work"


def make_atoms(coords, n):
    return Atoms(symbols=["N"] * n, positions=coords)


ENTRIES = []

# --- N4 ---
# 7, C2h, triplet, chaine ouverte (trans, planaire)
coords = geom.chain2d([1.176, 1.514, 1.176], [115.2, 115.2])
ENTRIES.append(dict(id="N4_C2h_chain_GS", formula="N4", charge=0, mult=3,
                     point_group="C2h", coords=coords, n=4,
                     method="Becke3LYP/6-311+G*"))

# 8, Td, tetraazatetrahedrane (arete unique)
coords = geom.tetrahedron(1.447)
ENTRIES.append(dict(id="N4_Td_GS", formula="N4", charge=0, mult=1,
                     point_group="Td", coords=coords, n=4,
                     method="Becke3LYP/6-311+G*"))

# 8a, D2h, tetrazete (cycle a 4, liaisons alternees)
coords = geom.rectangle(1.534, 1.249)
ENTRIES.append(dict(id="N4_D2h_tetrazete_GS", formula="N4", charge=0, mult=1,
                     point_group="D2h", coords=coords, n=4,
                     method="Becke3LYP/6-311+G*"))

# --- N6 ---
# 9, D6h, hexagone regulier -- SIGNALE COMME POINT-SELLE DU 2E ORDRE PAR [GS]
coords = geom.regular_ngon(6, 1.319)
ENTRIES.append(dict(id="N6_D6h_hexagon_GS", formula="N6", charge=0, mult=1,
                     point_group="D6h", coords=coords, n=6,
                     method="Becke3LYP/6-311+G*",
                     note="point-selle du 2e ordre d'apres [GS] (2 freq. imaginaires, e2u=272i cm-1)"))

# 10, D2, "twist-boat" (guess planaire initial -- pucker attendu a la relaxation)
coords, _ = geom.general_ring([1.326, 1.314, 1.326, 1.314, 1.326, 1.314],
                               [121.5, 115.4, 121.5, 115.4, 121.5, 115.4])
ENTRIES.append(dict(id="N6_D2_twistboat_GS", formula="N6", charge=0, mult=1,
                     point_group="D2", coords=coords, n=6,
                     method="Becke3LYP/6-311+G*"))

# 10a, C2h, chaine ouverte NNN-NNN (trans)
coords = geom.chain2d([1.133, 1.242, 1.442, 1.242, 1.133],
                       [109.5, 172.0, 172.0, 109.5])
ENTRIES.append(dict(id="N6_C2h_openchain_GS", formula="N6", charge=0, mult=1,
                     point_group="C2h", coords=coords, n=6,
                     method="Becke3LYP/6-311+G*"))


def main():
    results = []
    for e in ENTRIES:
        atoms = make_atoms(e["coords"], e["n"])
        run_dir = WORKDIR / e["id"]
        xyz_in = run_dir / "initial.xyz"
        run_dir.mkdir(parents=True, exist_ok=True)
        atoms.write(xyz_in)

        charge, uhf = e["charge"], e["mult"] - 1
        ok, e_ha, xyz_out = _tblite_optimize(
            str(xyz_in), run_dir, charge, uhf, gfn=2, opt_level="tight",
            max_steps=500, out_name="xtbopt.xyz")
        if not ok:
            print(f"{e['id']}: OPT FAILED")
            continue

        from ase.io import read
        relaxed = read(xyz_out)
        freq_dir = run_dir / "freq"
        result = verify_and_relax_to_minimum(relaxed, charge=charge, uhf=uhf, gfn=2,
                                              work_dir=freq_dir, max_hops=5)
        e_per_atom_ev = result["final_energy_ev"] / e["n"]
        results.append(dict(
            id=e["id"], formula=e["formula"], charge=charge, mult=e["mult"],
            point_group=e["point_group"], method=e["method"],
            is_true_minimum=result["is_minimum"], hops=result["hops_needed"],
            e_eV_per_atom=round(e_per_atom_ev, 6),
            note=e.get("note", ""),
        ))
        print(f"{e['id']}: minimum={result['is_minimum']} hops={result['hops_needed']} "
              f"E={e_per_atom_ev:.4f} eV/at")

    import csv
    out_path = ROOT / "gs_n4n6_results.csv"
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "formula", "charge", "mult", "point_group",
                                            "method", "is_true_minimum", "hops", "e_eV_per_atom", "note"])
        w.writeheader()
        for r in results:
            w.writerow(r)
    print("Ecrit:", out_path)


if __name__ == "__main__":
    main()
