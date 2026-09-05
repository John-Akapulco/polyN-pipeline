#!/usr/bin/env python3
"""
make_final_report_data.py
===========================

Consolide toutes les structures d'origine bibliographique (corpus initial
[GS]/[B], 70 structures + extension de 38 nouveaux articles) en un jeu de
donnees unique : nom, formule, charge, groupe ponctuel, energie GFN2-xTB
(eV/atome), reference article, energie relative (kcal/mol) au sein du
groupe (formule, charge). Ecrit un CSV consolide + les fragments de
tableaux LaTeX (un par etat de charge).
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent

_PG_RE = re.compile(r"^[CDSTOI][a-z0-9]*[hvd]?$", re.IGNORECASE)


def clean_pg(pg: str) -> str:
    """Ne garde que le symbole de groupe ponctuel (premier mot), remplace
    par 'n.d.' si l'article ne l'a pas donne numeriquement/explicitement."""
    first = pg.split()[0] if pg.split() else pg
    first = first.strip('"')
    if _PG_RE.match(first) and "stated" not in pg.lower():
        return first
    return "n.d."
HARTREE_TO_KCALMOL = 627.5094740631

# ---------------------------------------------------------------------
# 1. Corpus initial (70 structures, [GS]/[B])
# ---------------------------------------------------------------------
manifest = json.load(open(ROOT / "build" / "manifest.json"))
manifest_by_id = {e["id"]: e for e in manifest}

exact_energies = {}
with open(ROOT / "exact_structures_xtb_results.csv") as fh:
    for r in csv.DictReader(fh):
        if r["status"] == "ok":
            exact_energies[r["id"]] = float(r["e_xtb_eV_per_atom"])

topo2_energies = {}
with open(ROOT / "freq_verification_results_biblio.csv") as fh:
    for r in csv.DictReader(fh):
        if r["freq_status"] == "ok":
            topo2_energies[r["name"]] = float(r["e_final_eV_per_atom"])

REF_LABELS = {"[GS]": "GS", "[B]": "B"}
REF_CITATIONS = {
    "GS": "Glukhovtsev, Jiao, von Rag\\'e Schleyer, \\emph{Inorg. Chem.} \\textbf{1996}, 35, 7124--7133.",
    "B": "Fau, Mobita, Wilson, Perera, Bartlett, \\emph{Quantum Theory Project report}, University of Florida.",
}

records = []
for eid, e in manifest_by_id.items():
    if not re.fullmatch(r"N\d+", e["formula"]):
        continue  # exclut les especes non purement azotees (ex. N5H pentazole)
    energy = exact_energies.get(eid) if e["status"] == "exact" else topo2_energies.get(eid)
    if energy is None:
        continue
    tag = e["source"].split()[0]
    ref = REF_LABELS.get(tag, tag)
    records.append(dict(
        name=eid, formula=e["formula"], charge=e["charge"], n_atoms=e["n"],
        point_group=clean_pg(e["pg"]), e_eV_per_atom=energy, ref=ref,
        orig_method=e["method"],
    ))

# ---------------------------------------------------------------------
# 2. Extension archive (38 nouveaux articles) -- 3 nouvelles + 31 doublons,
#    hors les 8 "cages a anneaux imbriques" non fiables (N_nanotubes_06.json)
# ---------------------------------------------------------------------
catalog = {}
with open(ROOT / "archive_catalog.csv") as fh:
    for r in csv.DictReader(fh):
        catalog[r["filename"]] = r

ref_counter = {}
next_ref_num = 1


def ref_key_for(source_file: str) -> str:
    global next_ref_num
    if source_file not in ref_counter:
        ref_counter[source_file] = f"A{next_ref_num}"
        next_ref_num += 1
    return ref_counter[source_file]


def n_atoms_of(formula: str) -> int:
    m = re.match(r"N(\d+)", formula)
    return int(m.group(1)) if m else 0


with open(ROOT / "archive_new_structures_results.csv") as fh:
    for r in csv.DictReader(fh):
        if r["source_file"] == "N_nanotubes_06.json":
            continue  # famille "cages" exclue -- voir rapport section E
        if r["final_status"] not in ("NEW_UNIQUE_MINIMUM", "DUPLICATE_OF_EXISTING"):
            continue
        pdf_name = r["source_file"].replace(".json", ".pdf")
        cat = catalog.get(pdf_name, {})
        ref = ref_key_for(r["source_file"])
        REF_CITATIONS[ref] = (
            f"{cat.get('authors', '?')}, \\emph{{{cat.get('title', pdf_name)}}}, "
            f"{cat.get('journal_year', '?')}."
        )
        if not re.fullmatch(r"N\d+", r["formula"]):
            continue  # exclut les especes non purement azotees
        records.append(dict(
            name=r["id"], formula=r["formula"], charge=int(r["charge"]),
            n_atoms=n_atoms_of(r["formula"]), point_group=clean_pg(r["point_group"]),
            e_eV_per_atom=float(r["e_xtb_eV_per_atom"]), ref=ref,
            orig_method=r["method"],
        ))

# ---------------------------------------------------------------------
# 2bis. Complement [GS] : isomeres N4/N6 (Table 1/3, Fig. 1-2) manques a
#       l'extraction initiale. N6_D6h_hexagon_GS exclu : c'est un
#       point-selle du 2e ordre d'apres [GS] lui-meme, et notre correction
#       de frequence l'a fait basculer hors de la symetrie D6h -- son
#       energie finale ne represente plus la structure D6h annoncee.
# ---------------------------------------------------------------------
with open(ROOT / "gs_n4n6_results.csv") as fh:
    for r in csv.DictReader(fh):
        if r["id"] == "N6_D6h_hexagon_GS":
            continue
        if r["is_true_minimum"] != "True":
            continue
        records.append(dict(
            name=r["id"], formula=r["formula"], charge=int(r["charge"]),
            n_atoms=n_atoms_of(r["formula"]), point_group=clean_pg(r["point_group"]),
            e_eV_per_atom=float(r["e_eV_per_atom"]), ref="GS",
            orig_method=r["method"],
        ))

# ---------------------------------------------------------------------
# 2ter. Complements trouves lors de l'audit de completude des 38 articles
#       (entrees topology_only ecartees a tort du premier tri, reconstruites
#       depuis une connectivite confirmee visuellement sur une figure)
# ---------------------------------------------------------------------
reaudit_path = ROOT / "reaudit_new_structures.csv"
if reaudit_path.exists():
    with open(reaudit_path) as fh:
        for r in csv.DictReader(fh):
            pdf_name = r["source_file"].replace(".json", ".pdf")
            cat = catalog.get(pdf_name, {})
            ref = ref_key_for(r["source_file"])
            REF_CITATIONS[ref] = (
                f"{cat.get('authors', '?')}, \\emph{{{cat.get('title', pdf_name)}}}, "
                f"{cat.get('journal_year', '?')}."
            )
            records.append(dict(
                name=r["id"], formula=r["formula"], charge=int(r["charge"]),
                n_atoms=n_atoms_of(r["formula"]), point_group=clean_pg(r["point_group"]),
                e_eV_per_atom=float(r["e_eV_per_atom"]), ref=ref,
                orig_method=r["method"],
            ))

print(f"{len(records)} structures consolidees (energie GFN2-xTB + reference).")

# ---------------------------------------------------------------------
# 3. Energie relative par groupe (formule, charge), en kcal/mol / molecule
# ---------------------------------------------------------------------
groups = {}
for rec in records:
    groups.setdefault((rec["formula"], rec["charge"]), []).append(rec)

for (formula, charge), members in groups.items():
    e_min = min(r["e_eV_per_atom"] for r in members)
    for r in members:
        delta_ev_per_atom = r["e_eV_per_atom"] - e_min
        r["e_rel_kcalmol"] = round(delta_ev_per_atom * r["n_atoms"] * 23.060548, 2)

records.sort(key=lambda r: (r["charge"], r["n_atoms"], r["e_rel_kcalmol"]))

out_csv = ROOT / "final_report_consolidated.csv"
with open(out_csv, "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=["name", "formula", "charge", "n_atoms",
                                             "point_group", "e_eV_per_atom", "e_rel_kcalmol",
                                             "ref", "orig_method"])
    writer.writeheader()
    for r in records:
        writer.writerow(r)
print("Ecrit:", out_csv)

refs_path = ROOT / "final_report_references.json"
json.dump(REF_CITATIONS, open(refs_path, "w"), indent=1, ensure_ascii=False)
print("Ecrit:", refs_path, f"({len(REF_CITATIONS)} references)")
