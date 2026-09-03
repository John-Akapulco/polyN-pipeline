#!/usr/bin/env python3
"""
visualize_results.py
====================

Render the polynitrogen structures found by the pipeline as a 2D grid of
molecular depictions, each annotated with its name, charge family, stability
rank, and reaction/formation energy. Produces a PNG (and optionally a
multi-page PDF) for quick visual browsing of a run.

Two sources of 2D structure, in order of preference:
  1. the `smiles` column of results.csv, if present (newer runs) -- gives the
     cleanest depictions, since bonding/charges are explicit;
  2. otherwise, bonds are perceived from each .xyz geometry (older runs).
     Bond perception from 3D coordinates is heuristic for polynitrogen
     (multiple bonds, delocalized charges), so these depictions are
     approximate -- good enough to recognize a topology at a glance, but not
     a substitute for the real electronic structure.

Usage
-----
    python3 visualize_results.py --results resultats/seeds_pubchem/results.csv
    python3 visualize_results.py --results .../results.csv --family anion
    python3 visualize_results.py --results .../results.csv --per-composition 1
    python3 visualize_results.py --results .../results.csv --pdf

Options let you filter by family, cap how many ranks per composition to show
(e.g. only the most stable of each), and sort.

Dependencies
------------
    rdkit, pandas, matplotlib   (all already pipeline dependencies)
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


# ---------------------------------------------------------------------------
# Building an RDKit molecule for depiction
# ---------------------------------------------------------------------------

def mol_from_smiles(smi: str):
    """Clean 2D mol from a SMILES (preferred path)."""
    if not isinstance(smi, str) or not smi.strip():
        return None
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    AllChem.Compute2DCoords(m)
    return m


def mol_from_xyz(xyz_path, charge: int = 0):
    """Perceive a molecule (with bonds) from a 3D .xyz geometry.

    Handles two real-world issues:
      1. Non-standard comment line: some writers (e.g. tblite) put numeric
         data (gradient/dipole) on line 2 of the .xyz, which RDKit's
         MolFromXYZFile mis-parses as coordinates. We re-read the file
         manually and replace the comment line before parsing.
      2. Charge-aware bond perception: passing the known formal charge to
         DetermineBonds greatly improves bond-order/charge assignment for
         ions (the charge comes from the results.csv 'charge' column)."""
    if not xyz_path or not isinstance(xyz_path, str) or not xyz_path.strip():
        return None
    p = Path(xyz_path)
    if not p.exists():
        return None
    try:
        lines = p.read_text().splitlines()
        if len(lines) < 3:
            return None
        n_atoms = int(lines[0].split()[0])
        # Atom lines may be in ASE extended-xyz format, carrying extra columns
        # after x,y,z (initial_charges, forces, charge -- see the
        # "Properties=..." comment line). Keep ONLY symbol + the first three
        # numeric fields; otherwise RDKit tries to read the extra columns as
        # coordinates and fails ("Cannot convert ... to double").
        clean_atoms = []
        for ln in lines[2:2 + n_atoms]:
            parts = ln.split()
            if len(parts) < 4:
                continue
            sym = parts[0]
            x, y, z = parts[1], parts[2], parts[3]
            clean_atoms.append(f"{sym} {x} {y} {z}")
        if len(clean_atoms) != n_atoms:
            return None
        clean = f"{n_atoms}\n\n" + "\n".join(clean_atoms) + "\n"
        raw = Chem.MolFromXYZBlock(clean)
        if raw is None:
            return None
        mol = Chem.Mol(raw)
        try:
            from rdkit.Chem import rdDetermineBonds
            try:
                rdDetermineBonds.DetermineBonds(mol, charge=int(charge))
            except Exception:
                rdDetermineBonds.DetermineConnectivity(mol)
        except Exception:
            pass
        AllChem.Compute2DCoords(mol)
        return mol
    except Exception:
        return None


def get_mol(row):
    """Prefer SMILES if available, else fall back to xyz perception."""
    if "smiles" in row and isinstance(row.get("smiles"), str) and row["smiles"].strip():
        m = mol_from_smiles(row["smiles"])
        if m is not None:
            return m, "smiles"
    charge = int(row.get("charge", 0)) if not pd.isna(row.get("charge", 0)) else 0
    m = mol_from_xyz(row.get("xyz", ""), charge=charge)
    return m, "xyz"


# ---------------------------------------------------------------------------
# Annotation text
# ---------------------------------------------------------------------------

def make_legend(row) -> str:
    """One-line label under each molecule."""
    formula = row.get("formula", "?")
    family = row.get("family", "?")
    rank = row.get("rank", "?")
    e = row.get("e_reaction_kcalmol", None)
    fam_sym = {"neutral": "", "cation": "(+)", "anion": "(-)"}.get(family, "")
    e_txt = ""
    if e is not None and not (isinstance(e, float) and math.isnan(e)):
        e_txt = f"  {e:+.1f} kcal/mol"
    return f"{formula}{fam_sym}  #{rank}{e_txt}"


# ---------------------------------------------------------------------------
# Main rendering
# ---------------------------------------------------------------------------

def _draw_mol_svg(mol, size=320) -> str:
    """Render a single molecule to an SVG string (no Cairo needed)."""
    d = rdMolDraw2D.MolDraw2DSVG(size, size)
    d.DrawMolecule(mol)
    d.FinishDrawing()
    return d.GetDrawingText()


def _grid_via_matplotlib(mols, legends, out_path, mols_per_row, panel,
                         max_rows_per_page=6):
    """Assemble molecule depictions without Cairo. Paginates into multiple
    images of at most max_rows_per_page rows each, so no single image is
    unwieldy. Returns the list of written file paths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from io import BytesIO
    from PIL import Image
    import numpy as np

    rasterizer = None
    try:
        import cairosvg  # noqa
        rasterizer = "cairosvg"
    except Exception:
        try:
            from svglib.svglib import svg2rlg  # noqa
            from reportlab.graphics import renderPM  # noqa
            rasterizer = "svglib"
        except Exception:
            rasterizer = None

    per_page = mols_per_row * max_rows_per_page
    n_pages = math.ceil(len(mols) / per_page)
    written = []
    out_path = Path(out_path)
    stem, suffix = out_path.stem, out_path.suffix or ".png"

    for page in range(n_pages):
        chunk_m = mols[page * per_page:(page + 1) * per_page]
        chunk_l = legends[page * per_page:(page + 1) * per_page]
        n_rows = math.ceil(len(chunk_m) / mols_per_row)
        fig, axes = plt.subplots(n_rows, mols_per_row,
                                 figsize=(mols_per_row * 3.2, n_rows * 3.5),
                                 squeeze=False)
        axes = axes.flatten()
        for i, (mol, leg) in enumerate(zip(chunk_m, chunk_l)):
            ax = axes[i]
            drawn = False
            if rasterizer:
                try:
                    svg = _draw_mol_svg(mol, size=panel)
                    if rasterizer == "cairosvg":
                        import cairosvg
                        png = cairosvg.svg2png(bytestring=svg.encode(),
                                               output_width=panel, output_height=panel)
                    else:
                        from svglib.svglib import svg2rlg
                        from reportlab.graphics import renderPM
                        drawing = svg2rlg(BytesIO(svg.encode()))
                        png = renderPM.drawToString(drawing, fmt="PNG")
                    ax.imshow(np.asarray(Image.open(BytesIO(png))))
                    drawn = True
                except Exception:
                    drawn = False
            if not drawn:
                _draw_mol_matplotlib(ax, mol)
            ax.set_title(leg, fontsize=9)
            ax.axis("off")
        for j in range(len(chunk_m), len(axes)):
            axes[j].axis("off")
        fig.tight_layout()
        if n_pages == 1:
            page_path = out_path
        else:
            page_path = out_path.with_name(f"{stem}_p{page + 1:02d}{suffix}")
        fig.savefig(str(page_path), dpi=110, bbox_inches="tight")
        plt.close(fig)
        written.append(page_path)
    return written


def _draw_mol_matplotlib(ax, mol):
    """Last-resort depiction: draw the 2D skeleton with matplotlib only."""
    conf = mol.GetConformer()
    xs, ys = [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        pi, pj = conf.GetAtomPosition(i), conf.GetAtomPosition(j)
        order = b.GetBondTypeAsDouble()
        # draw multiple parallel lines for double/triple bonds
        n_lines = int(order) if order in (1, 2, 3) else 1
        for k in range(n_lines):
            off = (k - (n_lines - 1) / 2) * 0.08
            ax.plot([pi.x + off, pj.x + off], [pi.y + off, pj.y + off],
                    color="#1a3a8f", lw=1.4, zorder=1)
    for a in mol.GetAtoms():
        p = conf.GetAtomPosition(a.GetIdx())
        q = a.GetFormalCharge()
        label = a.GetSymbol() + ("" if q == 0 else ("+" if q > 0 else "\u2212"))
        ax.text(p.x, p.y, label, ha="center", va="center", fontsize=11,
                color="#1a3a8f", fontweight="bold", zorder=2,
                bbox=dict(boxstyle="round,pad=0.05", fc="white", ec="none"))
        xs.append(p.x); ys.append(p.y)
    if xs:
        m = 0.6
        ax.set_xlim(min(xs) - m, max(xs) + m)
        ax.set_ylim(min(ys) - m, max(ys) + m)
    ax.set_aspect("equal")


def render_grid(df: pd.DataFrame, out_path: Path, mols_per_row: int = 4,
                panel: int = 340, max_rows_per_page: int = 6) -> int:
    """Render all rows of df into one or more grid images, PAGINATED so no
    single image is unwieldy (204 molecules on one canvas is unreadable).
    Returns count drawn."""
    mols, legends = [], []
    n_from_smiles = n_from_xyz = n_failed = 0
    for _, row in df.iterrows():
        m, src = get_mol(row)
        if m is None:
            n_failed += 1
            continue
        mols.append(m)
        legends.append(make_legend(row))
        if src == "smiles":
            n_from_smiles += 1
        else:
            n_from_xyz += 1

    if not mols:
        print("No molecules could be rendered.")
        return 0

    # Always use the paginating matplotlib assembler (works with or without
    # Cairo, via the rasterizer probe inside it). This guarantees readable,
    # reasonably-sized pages regardless of how many structures there are.
    written = _grid_via_matplotlib(mols, legends, out_path, mols_per_row,
                                   panel, max_rows_per_page=max_rows_per_page)

    if len(written) == 1:
        print(f"Rendered {len(mols)} molecules -> {written[0]}")
    else:
        print(f"Rendered {len(mols)} molecules across {len(written)} pages:")
        for w in written:
            print(f"    {w}")
    print(f"  ({n_from_smiles} from SMILES, {n_from_xyz} from xyz geometry, "
          f"{n_failed} could not be drawn)")
    if n_from_xyz and not n_from_smiles:
        print("  NOTE: depictions perceived from 3D geometry are approximate "
              "for polynitrogen. Re-run the pipeline (updated code) for a "
              "'smiles' column and cleaner drawings.")
    return len(mols)


def main():
    ap = argparse.ArgumentParser(
        description="Render found polynitrogen structures as an annotated 2D grid.")
    ap.add_argument("--results", required=True, help="Path to results.csv")
    ap.add_argument("--out", default=None, help="Output image path (default: "
                    "alongside results.csv as structures_grid.png)")
    ap.add_argument("--family", choices=["neutral", "cation", "anion"], default=None,
                    help="Only this charge family.")
    ap.add_argument("--per-composition", type=int, default=None,
                    help="Keep at most this many ranks per (formula, family) "
                         "(e.g. 1 = only the most stable of each).")
    ap.add_argument("--min-true-minimum", action="store_true",
                    help="Only structures confirmed as true minima.")
    ap.add_argument("--mols-per-row", type=int, default=4)
    ap.add_argument("--rows-per-page", type=int, default=6,
                    help="Rows per image page (default 6; keeps each image "
                         "readable instead of one giant canvas).")
    ap.add_argument("--pdf", action="store_true",
                    help="Also write a multi-page PDF (one family per section).")
    args = ap.parse_args()

    results_path = Path(args.results)
    df = pd.read_csv(results_path)

    if args.family:
        df = df[df["family"] == args.family]
    if args.min_true_minimum and "is_true_minimum" in df.columns:
        df = df[df["is_true_minimum"] == True]  # noqa: E712
    if args.per_composition is not None and "rank" in df.columns:
        df = df[df["rank"] <= args.per_composition]

    if df.empty:
        print("No structures match the filters.")
        return

    # Sort for a sensible reading order: family, size, rank.
    sort_cols = [c for c in ["family", "n_atoms", "rank"] if c in df.columns]
    df = df.sort_values(sort_cols)

    out_path = Path(args.out) if args.out else results_path.parent / "structures_grid.png"
    render_grid(df, out_path, mols_per_row=args.mols_per_row,
                max_rows_per_page=args.rows_per_page)

    if args.pdf:
        pdf_path = out_path.with_suffix(".pdf")
        _render_pdf(df, pdf_path, args.mols_per_row)


def _render_pdf(df: pd.DataFrame, pdf_path: Path, mols_per_row: int):
    """Multi-page PDF, one section per family, via matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(str(pdf_path)) as pdf:
        for family in ["neutral", "cation", "anion"]:
            sub = df[df["family"] == family]
            if sub.empty:
                continue
            mols, legends = [], []
            for _, row in sub.iterrows():
                m, _ = get_mol(row)
                if m is not None:
                    mols.append(m)
                    legends.append(make_legend(row))
            if not mols:
                continue
            per_page = mols_per_row * 5
            for start in range(0, len(mols), per_page):
                chunk = mols[start:start + per_page]
                chunk_leg = legends[start:start + per_page]
                img = Draw.MolsToGridImage(
                    chunk, molsPerRow=mols_per_row, subImgSize=(300, 300),
                    legends=chunk_leg, useSVG=False)
                fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4
                ax.imshow(img)
                ax.axis("off")
                ax.set_title(f"{family.capitalize()} family "
                             f"(structures {start + 1}-{start + len(chunk)})",
                             fontsize=12)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
    print(f"Multi-page PDF -> {pdf_path}")


if __name__ == "__main__":
    main()
