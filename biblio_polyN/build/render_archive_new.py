# -*- coding: utf-8 -*-
"""Genere les figures ball-and-stick des structures issues de l'archive
complementaire (38 nouveaux articles) retenues pour le rapport."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
XYZDIR = os.path.join(ROOT, "archive_new_xyz")
FIGDIR = os.path.join(ROOT, "figures")

ENTRIES = [
    ("N18_C2v_cage_xtbopt.xyz", "N18 -- C2v cage (haute confiance)", "archive_N18_cage.png"),
    ("N7-_6_Cs_6ring_boat_xtbopt.xyz", "N7$^-$ -- Cs boat (confiance basse)", "archive_N7anion_boat.png"),
    ("N11_C2v_acyclic_chain_xtbopt.xyz", "N11 -- C2v chain (confiance basse)", "archive_N11_chain.png"),
]


def read_xyz(path):
    with open(path) as f:
        lines = f.readlines()
    n = int(lines[0].split()[0])
    syms, coords = [], []
    for line in lines[2:2 + n]:
        p = line.split()
        syms.append(p[0])
        coords.append([float(p[1]), float(p[2]), float(p[3])])
    return syms, np.array(coords)


def plot_one(xyz_name, label, out_name):
    syms, coords = read_xyz(os.path.join(XYZDIR, xyz_name))
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


if __name__ == "__main__":
    for xyz_name, label, out_name in ENTRIES:
        plot_one(xyz_name, label, out_name)
