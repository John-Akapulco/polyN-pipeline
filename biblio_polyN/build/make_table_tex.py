# -*- coding: utf-8 -*-
import json
import os

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)

with open(os.path.join(HERE, "manifest.json")) as f:
    rows = json.load(f)


def esc(s):
    return (s.replace("_", "\\_").replace("&", "\\&").replace("%", "\\%")
             .replace("#", "\\#"))


def charge_str(c):
    if c > 0:
        return "+" * c if c <= 2 else f"{c:+d}"
    if c < 0:
        return "-" * (-c) if -c <= 2 else f"{c:+d}"
    return ""


order = {"exact": 0, "xtb-refined": 1}
rows_sorted = sorted(rows, key=lambda r: (r["formula"].replace("N", "").zfill(3)
                                           if r["formula"][1:].isdigit() else r["formula"],
                                           order.get(r["status"], 2), r["id"]))

lines = []
lines.append(r"\begin{longtable}{@{}p{4.4cm}p{1.3cm}p{1.2cm}p{4.1cm}p{1.9cm}@{}}")
lines.append(r"\caption{Liste compl\`ete des 70 structures extraites de la "
             r"bibliographie, avec formule (charge en exposant), groupe "
             r"ponctuel, m\'ethode de la g\'eom\'etrie de r\'ef\'erence et "
             r"statut de reconstruction (exacte ou relax\'ee GFN2-xTB).}")
lines.append(r"\label{tab:structures}\\")
lines.append(r"\toprule")
lines.append(r"\textbf{Fichier (id)} & \textbf{Formule} & \textbf{G. ponctuel} & "
             r"\textbf{Methode} & \textbf{Statut} \\")
lines.append(r"\midrule")
lines.append(r"\endhead")
for r in rows_sorted:
    fid = esc(r["id"])
    formula = esc(r["formula"]) + ("$^{" + charge_str(r["charge"]) + "}$" if r["charge"] else "")
    pg = esc(r["pg"])
    method = esc(r["method"])
    status = "exacte" if r["status"] == "exact" else "GFN2-xTB"
    lines.append(f"\\texttt{{{fid}}} & {formula} & {pg} & {method} & {status} \\\\")
lines.append(r"\bottomrule")
lines.append(r"\end{longtable}")

with open(os.path.join(ROOT, "table_structures.tex"), "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"{len(rows)} lignes ecrites dans table_structures.tex")
