# -*- coding: utf-8 -*-
"""Genere les figures ball-and-stick des 7 structures gagnantes de la
comparaison a trois sources (biblio / N_csp / polyN_study), plus leur
comparateur biblio le plus stable du meme groupe (formule, charge), pour
l'annexe visuelle de rapport_polyN_biblio.tex."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from render import plot_one  # reutilise le renderer existant

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(ROOT)
FIGDIR = os.path.join(ROOT, "figures")

# (mol_id, xyz_path_absolu, label, out_name)
ENTRIES = [
    ("N4",  os.path.join(REPO, "structures_externes/xtb_relax/polyN_study/trial00102_N4_chain/xtbopt.xyz"),
     "N4 -- chain (polyN\\_study)", "win_N4_polyN_study.png"),
    ("N4b", os.path.join(ROOT, "xyz/N4_C2v_butterfly.xyz"),
     "N4 -- C2v papillon (biblio)", "win_N4_biblio.png"),

    ("N6",  os.path.join(REPO, "structures_externes/xtb_relax/N_csp/N6_0005_ring-3-substituted/xtbopt.xyz"),
     "N6 -- ring-3-subst. (N\\_csp)", "win_N6_N_csp.png"),
    ("N6b", os.path.join(ROOT, "xyz/N6_C2_book.xyz"),
     "N6 -- C2 book (biblio)", "win_N6_biblio.png"),

    ("N7",  os.path.join(REPO, "structures_externes/xtb_relax/N_csp/N7_0046_chain/xtbopt.xyz"),
     "N7 -- chain (N\\_csp)", "win_N7_N_csp.png"),
    ("N7b", os.path.join(ROOT, "xyz/N7_Cs_ring_chain.xyz"),
     "N7 -- Cs ring-chain (biblio)", "win_N7_biblio.png"),

    ("N8b", os.path.join(ROOT, "xyz/N8_C2v_ring.xyz"),
     "N8 -- C2v ring (biblio, gagnant)", "win_N8_biblio.png"),

    ("N9",  os.path.join(REPO, "structures_externes/xtb_relax/N_csp/N9_0042_ring-5-substituted/xtbopt.xyz"),
     "N9 -- ring-5-subst. (N\\_csp)", "win_N9_N_csp.png"),
    ("N9b", os.path.join(ROOT, "xyz/N9_fused_rings.xyz"),
     "N9 -- fused rings (biblio)", "win_N9_biblio.png"),

    ("N10", os.path.join(REPO, "structures_externes/xtb_relax/N_csp/N10_0070_ring-5-substituted/xtbopt.xyz"),
     "N10 -- ring-5-subst. (N\\_csp)", "win_N10_N_csp.png"),
    ("N10b", os.path.join(ROOT, "xyz/N10_C3_cap.xyz"),
     "N10 -- C3 cap (biblio)", "win_N10_biblio.png"),

    ("N12", os.path.join(REPO, "structures_externes/xtb_relax/polyN_study/trial00032_N12_ring-3-subst/xtbopt.xyz"),
     "N12 -- ring-3-subst. (polyN\\_study)", "win_N12_polyN_study.png"),
    ("N12b", os.path.join(ROOT, "xyz/N12_C2h_dipentazolyldiazene.xyz"),
     "N12 -- C2h dipentazolyldiazene (biblio)", "win_N12_biblio.png"),
]


def read_xyz_abs(path):
    with open(path) as f:
        lines = f.readlines()
    n = int(lines[0].split()[0] if lines[0].split() else lines[0])
    syms, coords = [], []
    for line in lines[2:2 + n]:
        p = line.split()
        syms.append(p[0])
        coords.append([float(p[1]), float(p[2]), float(p[3])])
    return syms, coords


if __name__ == "__main__":
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    for mol_id, xyz_path, label, out_name in ENTRIES:
        syms, coords = read_xyz_abs(xyz_path)
        coords = np.array(coords)
        fig = plt.figure(figsize=(3.2, 3.2))
        ax = fig.add_subplot(111, projection="3d")
        n = len(coords)
        D = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        for i in range(n):
            for j in range(i + 1, n):
                if D[i, j] < 1.9:
                    ax.plot(*zip(coords[i], coords[j]), color="#888888", lw=1.6, zorder=1)
        colors = ["#3060c0" if s == "N" else "#dddddd" for s in syms]
        sizes = [140 if s == "N" else 60 for s in syms]
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=colors, s=sizes,
                   edgecolor="k", linewidth=0.4, depthshade=True, zorder=2)
        ax.set_box_aspect([1, 1, 1])
        maxr = np.abs(coords - coords.mean(axis=0)).max() + 0.6
        c = coords.mean(axis=0)
        ax.set_xlim(c[0] - maxr, c[0] + maxr)
        ax.set_ylim(c[1] - maxr, c[1] + maxr)
        ax.set_zlim(c[2] - maxr, c[2] + maxr)
        ax.set_axis_off()
        ax.set_title(label, fontsize=8)
        fig.tight_layout(pad=0.2)
        outpath = os.path.join(FIGDIR, out_name)
        fig.savefig(outpath, dpi=200)
        plt.close(fig)
        print("figure ecrite:", outpath)
