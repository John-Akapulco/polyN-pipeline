#!/usr/bin/env python3
"""
make_bartlett_isodesmic.py
============================

Calcule l'enthalpie de formation quasi-isodesmique de chaque compose du
recueil (final_report_consolidated.csv), selon la methode de Fau, Wilson,
Perera, Bartlett ([B], polynitrogen1.pdf, p.75) :

  Neutres  : Nx -> (x/2) N2
             DHf(Nx) = E(Nx) - (x/2) E(N2)

  Cations  : [NH4]+ + (x-1)/2 N2 -> Nx+ + 2 H2         DHf(NH4+) = 150.6 kcal/mol
             DHf(Nx+) = DHf(NH4+) + [E(Nx+) + 2E(H2) - E(NH4+) - (x-1)/2 E(N2)]

  Anions   : [NH2]- + (x-1)/2 N2 -> Nx- + H2           DHf(NH2-) = 27.0 kcal/mol
             DHf(Nx-) = DHf(NH2-) + [E(Nx-) + E(H2) - E(NH2-) - (x-1)/2 E(N2)]

Toutes les energies electroniques sont GFN2-xTB (ce recueil), pas les
niveaux de calcul heterogenes des articles sources -- necessaire pour que
les DHf soient comparables entre composes.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
HARTREE_TO_KCALMOL = 627.5094740631
EV_TO_HARTREE = 1 / 27.211386245988

DHF_NH4_PLUS = 150.6   # kcal/mol, experimental (Michels, Montgomery, Christe, Dixon, JPCA 1995, 99, 187)
DHF_NH2_MINUS = 27.0   # kcal/mol, idem

ref_e = json.load(open(ROOT / "bartlett_reference_species" / "energies_hartree.json"))
E_NH4 = ref_e["NH4+"]
E_H2 = ref_e["H2"]
E_NH2 = ref_e["NH2-"]

with open(ROOT / "exact_structures_xtb_results.csv") as fh:
    E_N2 = next(float(r["e_xtb_hartree"]) for r in csv.DictReader(fh) if r["id"] == "N2")

rows = list(csv.DictReader(open(ROOT / "final_report_consolidated.csv")))

for r in rows:
    n = int(r["n_atoms"])
    charge = int(r["charge"])
    e_total_ha = float(r["e_eV_per_atom"]) * n * EV_TO_HARTREE
    if charge == 0:
        dhf = (e_total_ha - (n / 2) * E_N2) * HARTREE_TO_KCALMOL
    elif charge == 1:
        dhf = DHF_NH4_PLUS + (e_total_ha + 2 * E_H2 - E_NH4 - ((n - 1) / 2) * E_N2) * HARTREE_TO_KCALMOL
    elif charge == -1:
        dhf = DHF_NH2_MINUS + (e_total_ha + E_H2 - E_NH2 - ((n - 1) / 2) * E_N2) * HARTREE_TO_KCALMOL
    r["dHf_kcalmol"] = round(dhf, 1)

out_path = ROOT / "final_report_isodesmic.csv"
fieldnames = list(rows[0].keys())
with open(out_path, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
        w.writerow(r)
print(f"Ecrit: {out_path} ({len(rows)} composes)")
print(f"Reference N2 = {E_N2:.6f} Ha ; NH4+ = {E_NH4:.6f} Ha ; H2 = {E_H2:.6f} Ha ; NH2- = {E_NH2:.6f} Ha")
