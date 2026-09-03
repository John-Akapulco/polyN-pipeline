#!/usr/bin/env python3
"""
generate_report.py
==================

Build a structured LaTeX/PDF report of the polynitrogen structures retained
after all pipeline filters. The report is organized:

    Family (neutral -> anion -> cation)
      -> Nx cluster (by increasing size)
           -> every retained isomer, ground state first

For each isomer it lists:
  - the clean .xyz filename (VESTA-readable: plain "symbol x y z", no
    extended-xyz extra columns), written into a dedicated xyz_clean/ folder;
  - the relative energy (kcal/mol) with respect to the GROUND STATE of that
    Nx series (the most stable isomer of that size and family = 0);
  - the reaction energy (kcal/mol) as defined by the pipeline's per-family
    reference reaction (Nx -> (x/2) N2 for neutrals; Nx^q -> (x-5)/2 N2 +
    N5^q for ions), read from the results table;
  - a 2D depiction of the structure.

Outputs (next to the results CSV, or under --out-dir):
  - xyz_clean/<formula>_<family>_<rank>.xyz    (VESTA-ready)
  - report_images/<...>.png                    (2D depictions)
  - report_<author>.tex   and, if pdflatex is available, report_<author>.pdf

Usage
-----
    python3 generate_report.py --results resultats/seeds_pubchem/results_clean.csv \\
            --author "Gilles Frapper"
    python3 generate_report.py --results .../results_clean.csv --author Frapper \\
            --out-dir report_out --no-compile

Dependencies
------------
    rdkit, pandas, matplotlib   (pipeline deps); pdflatex for the PDF step.
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

HARTREE_TO_KCAL = 627.5094740631


# ---------------------------------------------------------------------------
# Clean XYZ (VESTA-readable)
# ---------------------------------------------------------------------------

def read_xyz_atoms(path: str):
    """Read atoms as (symbol, x, y, z) from a possibly extended-xyz file."""
    lines = Path(path).read_text().splitlines()
    n = int(lines[0].split()[0])
    atoms = []
    for ln in lines[2:2 + n]:
        p = ln.split()
        atoms.append((p[0], float(p[1]), float(p[2]), float(p[3])))
    return atoms


def write_clean_xyz(atoms, out_path: Path, comment: str = ""):
    """Write a standard, VESTA-readable .xyz (count / comment / 'S x y z')."""
    with open(out_path, "w") as f:
        f.write(f"{len(atoms)}\n{comment}\n")
        for s, x, y, z in atoms:
            f.write(f"{s:2s} {x:14.8f} {y:14.8f} {z:14.8f}\n")


# ---------------------------------------------------------------------------
# 2D depiction (reuses the robust xyz/SMILES perception approach)
# ---------------------------------------------------------------------------

def make_depiction(row, img_path: Path, panel: int = 360) -> bool:
    """Render a single 2D depiction PNG for a structure. Returns success."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    mol = None
    smi = row.get("smiles")
    if isinstance(smi, str) and smi.strip():
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            AllChem.Compute2DCoords(mol)
    if mol is None:
        # perceive from geometry
        try:
            atoms = read_xyz_atoms(str(row["xyz"]))
            n = len(atoms)
            block = f"{n}\n\n" + "\n".join(
                f"{s} {x} {y} {z}" for s, x, y, z in atoms) + "\n"
            raw = Chem.MolFromXYZBlock(block)
            if raw is not None:
                mol = Chem.Mol(raw)
                try:
                    from rdkit.Chem import rdDetermineBonds
                    charge = int(row.get("charge", 0)) if not pd.isna(row.get("charge", 0)) else 0
                    try:
                        rdDetermineBonds.DetermineBonds(mol, charge=charge)
                    except Exception:
                        rdDetermineBonds.DetermineConnectivity(mol)
                except Exception:
                    pass
                AllChem.Compute2DCoords(mol)
        except Exception:
            mol = None
    if mol is None:
        return False

    return _draw_png(mol, img_path, panel)


def _draw_png(mol, img_path: Path, panel: int) -> bool:
    """Draw a molecule to PNG, Cairo or Cairo-free."""
    try:
        from rdkit.Chem.Draw import rdMolDraw2D
        # Try Cairo first
        try:
            d = rdMolDraw2D.MolDraw2DCairo(panel, panel)
            d.DrawMolecule(mol)
            d.FinishDrawing()
            img_path.write_bytes(d.GetDrawingText())
            return True
        except Exception:
            pass
        # Cairo-free: matplotlib skeleton from 2D coords
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(3.2, 3.2))
        conf = mol.GetConformer()
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            pi, pj = conf.GetAtomPosition(i), conf.GetAtomPosition(j)
            order = b.GetBondTypeAsDouble()
            n_lines = int(order) if order in (1, 2, 3) else 1
            for k in range(n_lines):
                off = (k - (n_lines - 1) / 2) * 0.08
                ax.plot([pi.x + off, pj.x + off], [pi.y + off, pj.y + off],
                        color="#1a3a8f", lw=1.6, zorder=1)
        xs, ys = [], []
        for a in mol.GetAtoms():
            p = conf.GetAtomPosition(a.GetIdx())
            q = a.GetFormalCharge()
            lab = a.GetSymbol() + ("" if q == 0 else ("+" if q > 0 else "\u2212"))
            ax.text(p.x, p.y, lab, ha="center", va="center", fontsize=13,
                    color="#1a3a8f", fontweight="bold", zorder=2,
                    bbox=dict(boxstyle="round,pad=0.05", fc="white", ec="none"))
            xs.append(p.x); ys.append(p.y)
        if xs:
            m = 0.6
            ax.set_xlim(min(xs) - m, max(xs) + m)
            ax.set_ylim(min(ys) - m, max(ys) + m)
        ax.set_aspect("equal"); ax.axis("off")
        fig.savefig(str(img_path), dpi=120, bbox_inches="tight")
        plt.close(fig)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------

FAMILY_ORDER = ["neutral", "anion", "cation"]
FAMILY_TITLE = {"neutral": "Neutral", "anion": "Anionic", "cation": "Cationic"}
FAMILY_SIGN = {"neutral": "", "anion": r"$^{-}$", "cation": r"$^{+}$"}


def latex_escape(s: str) -> str:
    return (str(s).replace("\\", r"\textbackslash{}").replace("_", r"\_")
            .replace("%", r"\%").replace("&", r"\&").replace("#", r"\#"))


def build_latex(df: pd.DataFrame, author: str, img_dir: Path, xyz_dir: Path,
                out_dir: Path) -> str:
    """Assemble the full LaTeX source string."""
    author_esc = latex_escape(author)
    lines = [
        r"\documentclass[11pt,a4paper]{article}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[a4paper,margin=2.3cm]{geometry}",
        r"\usepackage{graphicx}",
        r"\usepackage{longtable}",
        r"\usepackage{booktabs}",
        r"\usepackage[table]{xcolor}",
        r"\usepackage{titlesec}",
        r"\usepackage{fancyhdr}",
        r"\usepackage{float}",
        r"\usepackage{hyperref}",
        r"\definecolor{accent}{HTML}{1F4E78}",
        r"\definecolor{gs}{HTML}{DDEBF7}",
        r"\hypersetup{colorlinks=true,linkcolor=accent,urlcolor=accent}",
        r"\titleformat{\section}{\color{accent}\Large\bfseries}{\thesection.}{0.5em}{}",
        r"\titleformat{\subsection}{\color{accent}\large\bfseries}{\thesubsection}{0.5em}{}",
        r"\pagestyle{fancy}\fancyhf{}",
        r"\renewcommand{\headrulewidth}{0.4pt}",
        r"\fancyhead[L]{\small\color{gray}Polynitrogen structures report}",
        r"\fancyhead[R]{\small\color{gray}" + author_esc + r"}",
        r"\fancyfoot[C]{\small\color{gray}\thepage}",
        r"\begin{document}",
        r"\begin{titlepage}\centering\vspace*{3cm}",
        r"{\Huge\bfseries\color{accent} Polynitrogen Allotropes\par}",
        r"\vspace{0.6em}{\Large Screened structures: energies and depictions\par}",
        r"\vspace{2em}{\large Neutral, anionic and cationic N$_x$ clusters\par}",
        r"\vspace{1em}{\large ground states and metastable isomers\par}",
        r"\vfill{\bfseries\Large " + author_esc + r"\par}",
        r"\vspace{0.5em}{\small IC2MP, UMR 7285 CNRS, University of Poitiers\par}",
        r"\vspace{1em}{\small\color{gray}Generated by the polyN pipeline\par}",
        r"\end{titlepage}",
        r"\tableofcontents\newpage",
        # Legend / methods note
        r"\section*{How to read this report}",
        r"\addcontentsline{toc}{section}{How to read this report}",
        r"Structures are grouped by charge family, then by cluster size "
        r"N$_x$. Within each series, isomers are ordered by increasing energy; "
        r"the most stable one (the \emph{ground state}, GS) is highlighted and "
        r"defines the zero of the \textbf{relative energy} $\Delta E_\text{rel}$ "
        r"(kcal/mol). The \textbf{reaction energy} $E_\text{react}$ (kcal/mol) "
        r"is computed against the pipeline's per-family reference reaction:",
        r"\begin{itemize}",
        r"\item neutral: $\mathrm{N}_x \rightarrow \tfrac{x}{2}\,\mathrm{N}_2$;",
        r"\item cation: $\mathrm{N}_x^{+} \rightarrow \tfrac{x-5}{2}\,\mathrm{N}_2 + \mathrm{N}_5^{+}$;",
        r"\item anion: $\mathrm{N}_x^{-} \rightarrow \tfrac{x-5}{2}\,\mathrm{N}_2 + \mathrm{N}_5^{-}$.",
        r"\end{itemize}",
        r"Each \texttt{.xyz} filename refers to a clean, VESTA-readable file in "
        r"the \texttt{xyz\_clean/} folder alongside this report. All energies "
        r"are at the GFN2-xTB level; structures are frequency-verified true "
        r"minima and passed the fragmentation filter.",
        r"\newpage",
    ]

    for family in FAMILY_ORDER:
        fam_df = df[df["family"] == family]
        if fam_df.empty:
            continue
        lines.append(r"\section{" + FAMILY_TITLE[family] + r" family}")
        # sizes present, increasing
        for n_atoms in sorted(fam_df["n_atoms"].unique()):
            series = fam_df[fam_df["n_atoms"] == n_atoms].copy()
            series = series.sort_values("rank")
            # ground-state energy for relative scale
            e_gs = series["e_xtb_hartree"].min()
            series["dE_rel"] = (series["e_xtb_hartree"] - e_gs) * HARTREE_TO_KCAL

            label = f"N$_{{{int(n_atoms)}}}${FAMILY_SIGN[family]}"
            lines.append(r"\subsection*{" + label + r"}")
            lines.append(r"\addcontentsline{toc}{subsection}{" +
                         f"N{int(n_atoms)} {FAMILY_TITLE[family]}" + r"}")

            # Table of isomers
            lines.append(r"\begin{longtable}{clrr}")
            lines.append(r"\toprule")
            lines.append(r"Rank & \texttt{.xyz} file & "
                         r"$\Delta E_\text{rel}$ & $E_\text{react}$ \\")
            lines.append(r" & & (kcal/mol) & (kcal/mol) \\")
            lines.append(r"\midrule\endhead")
            for _, r in series.iterrows():
                xyz_name = f"{r['formula']}_{family}_{int(r['rank']):03d}.xyz"
                de = r["dE_rel"]
                er = r.get("e_reaction_kcalmol", float("nan"))
                er_txt = f"{er:+.1f}" if pd.notna(er) else "--"
                gs_mark = r"\rowcolor{gs}" if int(r["rank"]) == 1 else ""
                gs_tag = r"~\textbf{(GS)}" if int(r["rank"]) == 1 else ""
                lines.append(
                    f"{gs_mark}{int(r['rank'])}{gs_tag} & "
                    f"\\texttt{{{latex_escape(xyz_name)}}} & "
                    f"{de:.1f} & {er_txt} \\\\")
            lines.append(r"\bottomrule")
            lines.append(r"\end{longtable}")

            # Depictions, a few per row via a figure
            imgs = []
            for _, r in series.iterrows():
                key = f"{r['formula']}_{family}_{int(r['rank']):03d}"
                png = img_dir / f"{key}.png"
                if png.exists():
                    imgs.append((int(r["rank"]), png))
            if imgs:
                lines.append(r"\begin{figure}[H]\centering")
                per_row = 4
                for idx, (rank, png) in enumerate(imgs):
                    rel = png.relative_to(out_dir)
                    lines.append(
                        r"\begin{minipage}{0.23\textwidth}\centering")
                    lines.append(
                        r"\includegraphics[width=\linewidth,height=3.2cm,"
                        r"keepaspectratio]{" + str(rel).replace("\\", "/") + r"}\\")
                    lines.append(r"{\scriptsize \#" + str(rank) + r"}")
                    lines.append(r"\end{minipage}")
                    if (idx + 1) % per_row == 0:
                        lines.append(r"\\[4pt]")
                lines.append(r"\end{figure}")
            lines.append(r"\clearpage")

    lines.append(r"\end{document}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(
        description="Generate a LaTeX/PDF report of retained polynitrogen "
                    "structures, grouped by family and Nx cluster.")
    ap.add_argument("--results", required=True,
                    help="Path to results CSV (ideally results_clean.csv).")
    ap.add_argument("--author", default="Gilles Frapper",
                    help="Author name (used in the title and filename).")
    ap.add_argument("--out-dir", default=None,
                    help="Output directory (default: next to the results CSV).")
    ap.add_argument("--no-compile", action="store_true",
                    help="Write the .tex but do not run pdflatex.")
    args = ap.parse_args()

    results_path = Path(args.results)
    df = pd.read_csv(results_path)

    out_dir = Path(args.out_dir) if args.out_dir else results_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    xyz_dir = out_dir / "xyz_clean"
    img_dir = out_dir / "report_images"
    xyz_dir.mkdir(exist_ok=True)
    img_dir.mkdir(exist_ok=True)

    # 1) clean xyz + 2D depictions for every retained isomer
    n_xyz = n_img = 0
    for _, r in df.iterrows():
        key = f"{r['formula']}_{r['family']}_{int(r['rank']):03d}"
        # clean xyz
        try:
            atoms = read_xyz_atoms(str(r["xyz"]))
            comment = (f"{r['formula']} {r['family']} rank {int(r['rank'])} "
                       f"E_react={r.get('e_reaction_kcalmol', float('nan')):.2f} kcal/mol")
            write_clean_xyz(atoms, xyz_dir / f"{key}.xyz", comment)
            n_xyz += 1
        except Exception:
            pass
        # depiction
        if make_depiction(r, img_dir / f"{key}.png"):
            n_img += 1
    print(f"Wrote {n_xyz} clean .xyz files -> {xyz_dir}")
    print(f"Wrote {n_img} depictions -> {img_dir}")

    # 2) LaTeX
    author_slug = args.author.split()[-1] if args.author.split() else "report"
    tex = build_latex(df, args.author, img_dir, xyz_dir, out_dir)
    tex_path = out_dir / f"report_{author_slug}.tex"
    tex_path.write_text(tex)
    print(f"Wrote LaTeX source -> {tex_path}")

    # 3) compile
    if args.no_compile:
        print("Skipping PDF compilation (--no-compile).")
        return
    if shutil.which("pdflatex") is None:
        print("pdflatex not found; wrote the .tex only. Compile it manually.")
        return
    for _ in range(2):  # twice for TOC
        subprocess.run(["pdflatex", "-interaction=nonstopmode",
                        tex_path.name], cwd=str(out_dir),
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    pdf_path = tex_path.with_suffix(".pdf")
    if pdf_path.exists():
        print(f"Report PDF -> {pdf_path}")
    else:
        print("pdflatex ran but no PDF was produced; check the .log in "
              f"{out_dir}")


if __name__ == "__main__":
    main()
