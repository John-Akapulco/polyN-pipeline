#!/usr/bin/env python3
"""Tableau comparatif : structures confirmees (final_report_consolidated.csv)
vs candidats identifies par recherche de citations (titre seul, gen1+2+3)."""
import json
import re
from collections import defaultdict
from pathlib import Path
import csv

ROOT = Path(__file__).parent.parent

# --- structures confirmees ---
confirmed = defaultdict(lambda: {"0": 0, "1": 0, "-1": 0})
for r in csv.DictReader(open(ROOT / "final_report_consolidated.csv")):
    confirmed[r["formula"]][r["charge"]] += 1

# --- candidats (titre seul) ---
gen1 = list(json.load(open(ROOT / "citation_search" / "pure_n_candidates_gen1.json")).values())
gen2 = json.load(open(ROOT / "citation_search" / "gen2_final_list.json"))
gen3 = json.load(open(ROOT / "citation_search" / "gen3_final_list.json"))
all_candidates = gen1 + gen2 + gen3

# Cas particuliers ou le parsing naif "N<chiffres><signe>" se trompe --
# corriges manuellement (charge multiple ecrite sans exposant, ex. N44- = N4(4-)).
MANUAL_OVERRIDES = {
    "Theoretical study on the two novel planar-type all-nitrogen N44": [("N4", "multi")],
}

FORMULA_RE = re.compile(r"\bN\s*(\d{1,2})")

def extract_formulas(title):
    for key_prefix, overrides in MANUAL_OVERRIDES.items():
        if title.startswith(key_prefix):
            return set(overrides)
    found = set()
    for m in re.finditer(FORMULA_RE, title):
        n = int(m.group(1))
        if n < 2 or n > 30:
            continue
        end = m.end()
        charge = "0"
        tail = title[end:end + 3]
        if re.match(r"\s*[+⁺]", tail):
            charge = "1"
        elif re.match(r"\s*[-⁻−](?!\w)", tail):
            charge = "-1"
        found.add((f"N{n}", charge))
    return found

candidates = defaultdict(lambda: {"0": 0, "1": 0, "-1": 0, "multi": 0})
unclear_count = 0
for w in all_candidates:
    formulas = extract_formulas(w["title"])
    if not formulas:
        unclear_count += 1
        continue
    for (f, c) in formulas:
        candidates[f][c] += 1

all_formulas = sorted(set(confirmed) | set(candidates), key=lambda f: int(f[1:]))


def esc(s):
    return s.replace("_", "\\_")


def cell(conf, cand):
    if conf == 0 and cand == 0:
        return "--"
    return f"{conf} / {cand}"


lines = [
    r"\begin{table}[H]",
    r"\centering",
    r"\small",
    r"\begin{tabular}{@{}lccc@{}}",
    r"\toprule",
    r"Formule & Neutre & Cation & Anion \\",
    r"\midrule",
]
for f in all_formulas:
    conf = confirmed.get(f, {"0": 0, "1": 0, "-1": 0})
    cand = candidates.get(f, {"0": 0, "1": 0, "-1": 0})
    n = f[1:]
    row = f"N$_{{{n}}}$ & {cell(conf['0'], cand['0'])} & {cell(conf['1'], cand['1'])} & {cell(conf['-1'], cand['-1'])}"
    extra = candidates.get(f, {}).get("multi", 0)
    if extra:
        row += f" \\quad {{\\tiny (+{extra} charge multiple, ex. N$_4^{{4-}}$)}}"
    lines.append(row + r" \\")
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(
    r"\caption{Structures confirm\'ees (construites, relax\'ees GFN2-xTB, "
    r"v\'erifi\'ees en fr\'equence) contre candidats identifi\'es par "
    r"recherche de citations (format : confirm\'ees / candidats). Les "
    r"candidats sont compt\'es \`a partir de la formule mentionn\'ee dans "
    r"le \textbf{titre seul} de l'article -- "
    + str(unclear_count)
    + r" des 158 candidats (49~\%) n'ont pas de formule identifiable dans "
    r"leur titre (revues, m\'ethodologie, commentaires) et n'apparaissent "
    r"donc pas dans cette table malgr\'e une pertinence potentielle. Un "
    r"compte de candidats n'implique aucune garantie de nouveaut\'e "
    r"structurale (un article peut tr\`es bien redocumenter une "
    r"topologie d\'ej\`a connue).}"
)
lines.append(r"\label{tab:confirmed-vs-candidates}")
lines.append(r"\end{table}")

(ROOT / "table_confirmed_vs_candidates.tex").write_text("\n".join(lines) + "\n")
print(f"table_confirmed_vs_candidates.tex written, {len(all_formulas)} formulas, "
      f"{unclear_count} candidates without identifiable formula")
