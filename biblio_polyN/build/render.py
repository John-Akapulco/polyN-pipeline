# -*- coding: utf-8 -*-
"""Genere de petites figures ball-and-stick (PNG) pour quelques structures
representatives, a inclure dans le rapport LaTeX."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
XYZDIR = os.path.join(ROOT, "xyz")
FIGDIR = os.path.join(ROOT, "figures")
os.makedirs(FIGDIR, exist_ok=True)

SELECTION = [
    ("N4_Td", "N4 (Td)"),
    ("N4_C2v_butterfly", "N4 (C2v, 'papillon')"),
    ("N5-_pentagon", "N5- (D5h, pentazolate)"),
    ("N5H_pentazole", "N5H (pentazole)"),
    ("N6_D3h_prism", "N6 (D3h, prisme)"),
    ("N8_Oh_cube", "N8 (Oh, cubane)"),
    ("N8_D2h_pentalene", "N8 (D2h, octaazapentalene)"),
    ("N10_D5h_prism", "N10 (D5h, prisme pentagonal)"),
    ("N10_D2d_linked5rings", "N10 (D2d, bispentazole)"),
    ("N20_Ih_dodecahedrane", "N20 (Ih, dodecaedrane)"),
]


def read_xyz(path):
    with open(path) as f:
        lines = f.readlines()
    n = int(lines[0])
    syms, coords = [], []
    for line in lines[2:2 + n]:
        p = line.split()
        syms.append(p[0])
        coords.append([float(p[1]), float(p[2]), float(p[3])])
    return syms, np.array(coords)


def plot_one(mol_id, label, outpath):
    syms, coords = read_xyz(os.path.join(XYZDIR, mol_id + ".xyz"))
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
    ax.set_title(label, fontsize=9)
    fig.tight_layout(pad=0.2)
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    for mol_id, label in SELECTION:
        out = os.path.join(FIGDIR, mol_id + ".png")
        plot_one(mol_id, label, out)
        print("figure ecrite:", out)
