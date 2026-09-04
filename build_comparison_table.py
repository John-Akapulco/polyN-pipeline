#!/usr/bin/env python3
"""
build_comparison_table.py
==========================

Assemble le tableau comparatif a trois sources demande : pour chaque
isomere N_n^q (formule + charge), classement par energie GFN2-xTB (du plus
stable au moins stable), avec l'origine (article biblio / import N_csp /
import polyN_study).

Sources
-------
- biblio_polyN/xtb_work/*/xtbopt.log : structures extraites des deux
  articles de reference, deja relaxees (CLI xtb, GFN2). Charge lue depuis
  le suffixe du nom de dossier (+/-), verifiee par la somme des charges
  partielles (fichier `charges`).
- structures_externes/xtb_relax_results.csv : structures importees
  (N_csp, polyN_study), relaxees ici meme (tblite, GFN2) -- toutes neutres.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

ROOT = Path(__file__).parent
HARTREE_TO_EV = 27.211386245988


def parse_biblio() -> list[dict]:
    rows = []
    for d in sorted((ROOT / "biblio_polyN" / "xtb_work").iterdir()):
        log = d / "xtbopt.log"
        charges_f = d / "charges"
        if not log.exists():
            continue
        lines = log.read_text().splitlines()
        n_atoms = int(lines[0].strip())
        last_e = None
        for line in reversed(lines):
            m = re.search(r"energy:\s*(-?\d+\.\d+)", line)
            if m:
                last_e = float(m.group(1))
                break
        if last_e is None:
            continue
        charge = 0
        if charges_f.exists():
            vals = [float(x) for x in charges_f.read_text().split()]
            charge = round(sum(vals))
        name = d.name
        m = re.match(r"^(N\d+)([+-]?)_(.+)$", name)
        if m:
            formula, sign, topo = m.groups()
        else:
            formula, topo = f"N{n_atoms}", name
        rows.append({
            "name": name,
            "origin": "biblio_article",
            "formula": formula,
            "n_atoms": n_atoms,
            "charge": charge,
            "topology_label": topo,
            "e_xtb_hartree": last_e,
        })
    return rows


def parse_imports() -> list[dict]:
    rows = []
    path = ROOT / "structures_externes" / "xtb_relax_results.csv"
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["converged"] != "True":
                continue
            n_atoms = int(row["n_atoms"])
            rows.append({
                "name": row["tag"],
                "origin": f"import_{row['source']}",
                "formula": f"N{n_atoms}",
                "n_atoms": n_atoms,
                "charge": int(row["charge"]),
                "topology_label": row["topology_label"],
                "e_xtb_hartree": float(row["e_xtb_hartree"]),
            })
    return rows


def main():
    all_rows = parse_biblio() + parse_imports()
    print(f"Total structures dans le tableau : {len(all_rows)} "
          f"({sum(1 for r in all_rows if r['origin']=='biblio_article')} biblio, "
          f"{sum(1 for r in all_rows if r['origin']=='import_N_csp')} N_csp, "
          f"{sum(1 for r in all_rows if r['origin']=='import_polyN_study')} polyN_study)")

    for r in all_rows:
        r["e_eV_per_atom"] = r["e_xtb_hartree"] / r["n_atoms"] * HARTREE_TO_EV

    groups: dict[tuple, list[dict]] = {}
    for r in all_rows:
        groups.setdefault((r["formula"], r["charge"]), []).append(r)

    out_rows = []
    for (formula, charge), members in groups.items():
        members.sort(key=lambda r: r["e_eV_per_atom"])
        best = members[0]["e_eV_per_atom"]
        for rank, r in enumerate(members, 1):
            r["rank_in_group"] = rank
            r["delta_eV_per_atom_vs_group_best"] = round(r["e_eV_per_atom"] - best, 5)
            r["e_eV_per_atom"] = round(r["e_eV_per_atom"], 5)
            out_rows.append(r)

    out_rows.sort(key=lambda r: (r["formula"], r["charge"], r["rank_in_group"]))

    out_path = ROOT / "biblio_polyN" / "comparison_table_3sources.csv"
    fieldnames = ["formula", "charge", "rank_in_group", "name", "topology_label",
                  "origin", "n_atoms", "e_xtb_hartree", "e_eV_per_atom",
                  "delta_eV_per_atom_vs_group_best"]
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in out_rows:
            writer.writerow({k: r[k] for k in fieldnames})
    print(f"Ecrit: {out_path}")

    n_groups_multi = sum(1 for m in groups.values() if len(m) > 1)
    n_groups_mixed_origin = sum(
        1 for m in groups.values()
        if len(set(r["origin"] for r in m)) > 1
    )
    print(f"Groupes (formule, charge) : {len(groups)} total, "
          f"{n_groups_multi} avec >1 structure, "
          f"{n_groups_mixed_origin} avec origines melangees (comparaison directe possible)")

    print("\n--- Rang 1 (le plus stable) par groupe avec origines melangees ---")
    for (formula, charge), members in sorted(groups.items()):
        origins = set(r["origin"] for r in members)
        if len(origins) <= 1:
            continue
        members.sort(key=lambda r: r["e_eV_per_atom"])
        winner = members[0]
        sign = {1: "+", -1: "-", 0: ""}.get(charge, str(charge))
        print(f"  {formula}{sign} ({len(members)} structures, origines={origins}): "
              f"gagnant = {winner['name']} [{winner['origin']}] "
              f"topo={winner['topology_label']} E={winner['e_eV_per_atom']:.4f} eV/atom")


if __name__ == "__main__":
    main()
