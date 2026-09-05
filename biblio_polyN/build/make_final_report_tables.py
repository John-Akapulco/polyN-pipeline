#!/usr/bin/env python3
"""Genere les 3 longtables (neutre/cation/anion) + la bibliographie en
.tex a partir de final_report_consolidated.csv / final_report_references.json."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent


def esc(s):
    return s.replace("_", "\\_").replace("%", "\\%")


def pg_tex(pg):
    return esc(pg)


rows = list(csv.DictReader(open(ROOT / "final_report_consolidated.csv")))
refs = json.load(open(ROOT / "final_report_references.json"))

LOW_CONFIDENCE_IDS = {"N7-_6_Cs_6ring_boat", "N11_C2v_acyclic_chain"}

groups = {"0": [], "1": [], "-1": []}
for r in rows:
    groups[r["charge"]].append(r)


def formula_tex(formula, charge):
    n = formula[1:]
    sup = ""
    if charge == "1":
        sup = "^{+}"
    elif charge == "-1":
        sup = "^{-}"
    return f"N$_{{{n}}}{sup}$"


CAPTIONS = {
    "0": ("Compos\\'es polyazot\\'es \\textbf{neutres} document\\'es dans la "
          "bibliographie : nom, groupe ponctuel, \\'energie GFN2-xTB (\\'eV/atome), "
          "\\'energie relative au sein du groupe (formule) en kcal/mol, r\\'ef\\'erence.",
          "tab:final-neutre"),
    "1": ("Compos\\'es polyazot\\'es \\textbf{cationiques} document\\'es dans la "
          "bibliographie.", "tab:final-cation"),
    "-1": ("Compos\\'es polyazot\\'es \\textbf{anioniques} document\\'es dans la "
           "bibliographie.", "tab:final-anion"),
}

for charge, fname in [("0", "table_neutre.tex"), ("1", "table_cation.tex"), ("-1", "table_anion.tex")]:
    members = groups[charge]
    members.sort(key=lambda r: (int(r["n_atoms"]), float(r["e_rel_kcalmol"])))
    caption, label = CAPTIONS[charge]
    lines = []
    lines.append(r"\begin{longtable}{@{}p{4.4cm}lp{1.5cm}rrc@{}}")
    lines.append(r"\caption{" + caption + r"}")
    lines.append(r"\label{" + label + r"}\\")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Nom} & \textbf{Formule} & \textbf{Sym.} & "
                  r"\textbf{$E$ (eV/at.)} & \textbf{$E_{\mathrm{rel}}$ (kcal/mol)} & \textbf{R\'ef.} \\")
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\multicolumn{6}{c}{\small (suite)}\\")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Nom} & \textbf{Formule} & \textbf{Sym.} & "
                  r"\textbf{$E$ (eV/at.)} & \textbf{$E_{\mathrm{rel}}$ (kcal/mol)} & \textbf{R\'ef.} \\")
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append(r"\bottomrule")
    lines.append(r"\endfoot")
    lines.append(r"\bottomrule")
    lines.append(r"\endlastfoot")
    prev_formula = None
    for r in members:
        if r["formula"] != prev_formula:
            lines.append(r"\addlinespace")
            prev_formula = r["formula"]
        name = esc(r["name"])
        marker = "$^{\\dagger}$" if r["name"] in LOW_CONFIDENCE_IDS else ""
        formula = formula_tex(r["formula"], charge)
        pg = pg_tex(r["point_group"])
        e = f"{float(r['e_eV_per_atom']):.4f}".replace(".", "{,}")
        erel = f"{float(r['e_rel_kcalmol']):.2f}".replace(".", "{,}")
        ref = r["ref"]
        lines.append(f"\\texttt{{{name}}}{marker} & {formula} & {pg} & {e} & {erel} & [{ref}] \\\\")
    if any(r["name"] in LOW_CONFIDENCE_IDS for r in members):
        lines.append(r"\multicolumn{6}{@{}p{15.5cm}}{\footnotesize $^{\dagger}$ Reconstruction "
                      r"de confiance basse (repli g\'en\'erique faute de connectivit\'e "
                      r"compl\`ete dans l'article source) -- minimum GFN2-xTB v\'erifi\'e, mais "
                      r"rien ne garantit la co\"incidence avec l'isom\`ere pr\'ecis d\'ecrit dans "
                      r"l'article.}\\")
    lines.append(r"\end{longtable}")
    (ROOT / fname).write_text("\n".join(lines) + "\n")
    print(f"{fname}: {len(members)} lignes")

# References section
lines = [r"\begin{description}[leftmargin=1.6cm,itemsep=3pt,style=nextline]"]
ref_order = ["GS", "B"] + sorted([k for k in refs if k not in ("GS", "B")],
                                  key=lambda k: int(k[1:]))
for k in ref_order:
    lines.append(f"\\item[{esc('['+k+']')}] {refs[k]}")
lines.append(r"\end{description}")
(ROOT / "table_references.tex").write_text("\n".join(lines) + "\n")
print(f"table_references.tex: {len(ref_order)} references")
