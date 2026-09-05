#!/usr/bin/env python3
"""Genere les 3 tableaux d'enthalpie de reaction quasi-isodesmique
(neutre/cation/anion) a partir de final_report_isodesmic.csv."""
import csv

from pathlib import Path

ROOT = Path(__file__).parent.parent


def esc(s):
    return s.replace("_", "\\_").replace("%", "\\%")


rows = list(csv.DictReader(open(ROOT / "final_report_isodesmic.csv")))
groups = {"0": [], "1": [], "-1": []}
for r in rows:
    groups[r["charge"]].append(r)

REACTIONS = {
    "0": "N$_x$ $\\rightarrow$ (x/2) N$_2$",
    "1": "[NH$_4$]$^+$ + (x$-$1)/2 N$_2$ $\\rightarrow$ N$_x^+$ + 2 H$_2$",
    "-1": "[NH$_2$]$^-$ + (x$-$1)/2 N$_2$ $\\rightarrow$ N$_x^-$ + H$_2$",
}
CAPTIONS = {
    "0": ("Enthalpie de r\\'eaction quasi-isodesmique (GFN2-xTB) des compos\\'es "
          "\\textbf{neutres}, r\\'eaction N$_x \\to$ (x/2)~N$_2$.", "tab:isodesmic-neutre"),
    "1": ("Enthalpie de r\\'eaction quasi-isodesmique (GFN2-xTB) des compos\\'es "
          "\\textbf{cationiques}, r\\'ef\\'erenc\\'ee \\`a NH$_4^+$ "
          "($\\Delta H_f=150{,}6$ kcal/mol, valeur exp\\'erimentale).", "tab:isodesmic-cation"),
    "-1": ("Enthalpie de r\\'eaction quasi-isodesmique (GFN2-xTB) des compos\\'es "
           "\\textbf{anioniques}, r\\'ef\\'erenc\\'ee \\`a NH$_2^-$ "
           "($\\Delta H_f=27{,}0$ kcal/mol, valeur exp\\'erimentale).", "tab:isodesmic-anion"),
}


def formula_tex(formula, charge):
    n = formula[1:]
    sup = "^{+}" if charge == "1" else "^{-}" if charge == "-1" else ""
    return f"N$_{{{n}}}{sup}$"


for charge, fname in [("0", "table_isodesmic_neutre.tex"),
                       ("1", "table_isodesmic_cation.tex"),
                       ("-1", "table_isodesmic_anion.tex")]:
    members = groups[charge]
    members.sort(key=lambda r: (int(r["n_atoms"]), float(r["dHf_kcalmol"])))
    caption, label = CAPTIONS[charge]
    lines = [
        r"\begin{longtable}{@{}p{4.6cm}lp{1.4cm}rc@{}}",
        r"\caption{" + caption + r"}",
        r"\label{" + label + r"}\\",
        r"\toprule",
        r"\textbf{Nom} & \textbf{Formule} & \textbf{Sym.} & \textbf{$\Delta H_f$ (kcal/mol)} & \textbf{R\'ef.} \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{5}{c}{\small (suite)}\\",
        r"\toprule",
        r"\textbf{Nom} & \textbf{Formule} & \textbf{Sym.} & \textbf{$\Delta H_f$ (kcal/mol)} & \textbf{R\'ef.} \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    prev_formula = None
    for r in members:
        if r["formula"] != prev_formula:
            lines.append(r"\addlinespace")
            prev_formula = r["formula"]
        name = esc(r["name"])
        formula = formula_tex(r["formula"], charge)
        pg = esc(r["point_group"])
        dhf = f"{float(r['dHf_kcalmol']):.1f}".replace(".", "{,}")
        ref = r["ref"]
        lines.append(f"\\texttt{{{name}}} & {formula} & {pg} & {dhf} & [{ref}] \\\\")
    lines.append(r"\end{longtable}")
    (ROOT / fname).write_text("\n".join(lines) + "\n")
    print(f"{fname}: {len(members)} lignes, reaction {REACTIONS[charge]}")
