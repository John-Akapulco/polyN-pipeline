#!/usr/bin/env python3
"""Genere le tableau LaTeX des candidats identifies par recherche de
citations (OpenAlex, generation 1), tries par annee."""
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent


def esc(s):
    return (s.replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")
             .replace("#", "\\#"))


pure_n = json.load(open(ROOT / "citation_search" / "pure_n_candidates_gen1.json"))
rows = sorted(pure_n.values(), key=lambda w: (w["year"] or 0, w["title"]))

lines = [
    r"\begin{longtable}{@{}p{1.0cm}p{10.5cm}p{3.5cm}@{}}",
    r"\caption{Candidats identifi\'es par recherche de citations (OpenAlex, "
    r"g\'en\'eration 1 : articles citant les 38 sources de ce recueil), "
    r"apr\`es filtrage strict aux compos\'es exclusivement azot\'es (neutres "
    r"ou charg\'es) -- 135 articles sur 977 citations uniques recens\'ees.}",
    r"\label{tab:citation-search-candidates}\\",
    r"\toprule",
    r"\textbf{Ann\'ee} & \textbf{Titre} & \textbf{DOI} \\",
    r"\midrule",
    r"\endfirsthead",
    r"\multicolumn{3}{c}{\small (suite)}\\",
    r"\toprule",
    r"\textbf{Ann\'ee} & \textbf{Titre} & \textbf{DOI} \\",
    r"\midrule",
    r"\endhead",
    r"\bottomrule",
    r"\endfoot",
    r"\bottomrule",
    r"\endlastfoot",
]
prev_year = None
for w in rows:
    y = w["year"] or "?"
    if y != prev_year:
        lines.append(r"\addlinespace")
        prev_year = y
    title = esc(w["title"])
    doi = w.get("doi") or ""
    doi = doi.replace("https://doi.org/", "") if doi else "--"
    lines.append(f"{y} & {title} & \\small\\texttt{{{esc(doi)}}} \\\\")
lines.append(r"\end{longtable}")

(ROOT / "table_citation_search.tex").write_text("\n".join(lines) + "\n")
print(f"table_citation_search.tex: {len(rows)} lignes")
