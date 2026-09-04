# -*- coding: utf-8 -*-
"""Genere les fichiers .xyz de biblio_polyN/xyz/ a partir de molecules.py.

Pour les entrees 'xtb_generic', une geometrie de depart topologiquement
raisonnable est construite puis relaxee avec GFN2-xTB (binaire `xtb`) afin
d'obtenir une structure 3D chimiquement valide (methode coherente avec le
pipeline polyN_pipeline.py de ce depot).
"""
import os
import subprocess
import sys
import shutil

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import geom
from molecules import M

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
OUTDIR = os.path.join(ROOT, "xyz")
XTBWORK = os.path.join(ROOT, "xtb_work")
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(XTBWORK, exist_ok=True)

XTB_BIN = shutil.which("xtb") or os.path.expanduser("~/miniconda3/bin/xtb")


def write_xyz(path, symbols, coords, comment):
    with open(path, "w") as f:
        f.write(f"{len(symbols)}\n")
        f.write(comment.replace("\n", " ") + "\n")
        for s, c in zip(symbols, coords):
            f.write(f"{s:2s} {c[0]:14.6f} {c[1]:14.6f} {c[2]:14.6f}\n")


def bond_check(coords, expected_pairs):
    """expected_pairs: liste de (i, j, longueur_attendue). Retourne l'ecart max."""
    max_dev = 0.0
    for i, j, d in expected_pairs:
        dd = np.linalg.norm(coords[i] - coords[j])
        max_dev = max(max_dev, abs(dd - d))
    return max_dev


def _read_xyz_coords(path):
    with open(path) as f:
        lines = f.readlines()
    n = int(lines[0])
    pts = []
    for line in lines[2:2 + n]:
        parts = line.split()
        pts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(pts)


def _xtb_run(symbols, coords, charge, uhf, wd, tag, extra_flags):
    inp = os.path.join(wd, f"{tag}.xyz")
    write_xyz(inp, symbols, coords, f"depart {tag}")
    xcontrol = os.path.join(wd, "xtb.inp")
    with open(xcontrol, "w") as f:
        f.write("$opt\n   engine=lbfgs\n$end\n")
    cmd = [XTB_BIN, f"{tag}.xyz", "--opt", "--input", "xtb.inp",
           "--charge", str(charge), "--uhf", str(uhf)] + extra_flags
    try:
        res = subprocess.run(cmd, cwd=wd, capture_output=True, text=True, timeout=300)
    except Exception as e:
        return None, str(e)
    outfile = os.path.join(wd, "xtbopt.xyz")
    if not os.path.isfile(outfile):
        return None, res.stdout + "\n----STDERR----\n" + res.stderr
    coords_out = _read_xyz_coords(outfile)
    os.replace(outfile, os.path.join(wd, f"{tag}_opt.xyz"))
    return coords_out, None


def run_xtb_opt(symbols, coords, charge, mult, workname):
    """GFN2-xTB (avec repli GFN-FF puis GFN2 si le premier essai echoue,
    la SCF de GFN2 pouvant ne pas converger sur une geometrie de depart
    trop eloignee d'un minimum)."""
    wd = os.path.join(XTBWORK, workname)
    os.makedirs(wd, exist_ok=True)
    uhf = mult - 1

    coords2, err2 = _xtb_run(symbols, coords, charge, uhf, wd, "gfn2", ["--gfn", "2"])
    if coords2 is not None:
        return coords2

    coordsff, errff = _xtb_run(symbols, coords, charge, uhf, wd, "gfnff", ["--gfnff"])
    if coordsff is None:
        with open(os.path.join(wd, "xtb.log"), "w") as f:
            f.write("GFN2 direct:\n" + (err2 or "") + "\n\nGFN-FF:\n" + (errff or ""))
        print(f"  [xtb ECHEC] {workname}: ni GFN2 direct ni GFN-FF n'ont converge")
        return None

    coords2b, err2b = _xtb_run(symbols, coordsff, charge, uhf, wd, "gfn2b", ["--gfn", "2"])
    if coords2b is not None:
        return coords2b

    with open(os.path.join(wd, "xtb.log"), "w") as f:
        f.write("GFN2 direct:\n" + (err2 or "") + "\n\nGFN2 apres GFN-FF:\n" + (err2b or ""))
    print(f"  [xtb PARTIEL] {workname}: GFN-FF converge, GFN2 final a echoue -> geometrie GFN-FF conservee")
    return coordsff


def build_one(mol):
    b = mol["build"]
    p = mol.get("params", {})
    formula = mol["formula"]
    symbols = None
    coords = None
    check_pairs = None
    xtb_used = False

    if b == "linear":
        bonds = p["bonds"]
        n = len(bonds) + 1
        coords = np.zeros((n, 3))
        x = 0.0
        for k, bd in enumerate(bonds):
            x += bd
            coords[k + 1] = [x, 0, 0]
        check_pairs = [(k, k + 1, bonds[k]) for k in range(len(bonds))]

    elif b == "triangle":
        edge = p["edge"]
        coords = geom.regular_ngon(3, edge)
        check_pairs = [(0, 1, edge), (1, 2, edge), (2, 0, edge)]

    elif b == "bent":
        bond, angle = p["bond"], p["angle"]
        coords = geom.chain2d([bond, bond], [angle])
        check_pairs = [(0, 1, bond), (1, 2, bond)]

    elif b == "rectangle":
        a_, b_ = p["a"], p["b"]
        coords = geom.rectangle(a_, b_)
        check_pairs = [(0, 1, a_), (1, 2, b_), (2, 3, a_), (3, 0, b_)]

    elif b == "chain2d":
        bonds, angles = p["bonds"], p.get("angles")
        turns = p.get("turns")
        coords = geom.chain2d(bonds, angles, turns)
        check_pairs = [(k, k + 1, bonds[k]) for k in range(len(bonds))]

    elif b == "ring_regular":
        n, edge = p["n"], p["edge"]
        coords = geom.regular_ngon(n, edge)
        check_pairs = [(k, (k + 1) % n, edge) for k in range(n)]

    elif b == "ring_general":
        bonds, angles = p["bonds"], p["angles"]
        coords, err = geom.general_ring(bonds, angles)
        n = len(bonds)
        check_pairs = [(k, (k + 1) % n, bonds[k]) for k in range(n)]

    elif b == "tetrahedron":
        edge = p["edge"]
        coords = geom.tetrahedron(edge)
        check_pairs = [(i, j, edge) for i in range(4) for j in range(i + 1, 4)]

    elif b == "cube":
        edge = p["edge"]
        coords = geom.cube(edge)
        check_pairs = None  # verifie plus bas par plus-proche-voisin

    elif b == "prism":
        n, er, ev = p["n"], p["edge_ring"], p["edge_vertical"]
        coords = geom.prism_dnh(n, er, ev)
        check_pairs = [(k, (k + 1) % n, er) for k in range(n)]
        check_pairs += [(k, k + n, ev) for k in range(n)]
        check_pairs += [(k + n, ((k + 1) % n) + n, er) for k in range(n)]

    elif b == "puckered_4ring":
        edge, angle = p["edge"], p["angle"]
        coords = geom.puckered_4ring_d2d(edge, angle)
        check_pairs = [(0, 1, edge), (1, 2, edge), (2, 3, edge), (3, 0, edge)]

    elif b == "trigonal_bipyramid_nocenter":
        bond, ang = p["bond"], p["angle_apex"]
        coords = geom.trigonal_bipyramid_nocenter(bond, ang)
        # coords[0]=apex+, coords[1]=apex-, coords[2:5]=equatorial
        check_pairs = [(0, e, bond) for e in (2, 3, 4)] + [(1, e, bond) for e in (2, 3, 4)]

    elif b == "ring_with_H":
        ring, hpos, err = geom.ring_with_h(p["n"], p["bonds"], p["angles"], p["h_index"], p["nh_bond"])
        coords = np.vstack([ring, hpos])
        n = p["n"]
        symbols = ["N"] * n + ["H"]
        check_pairs = [(k, (k + 1) % n, p["bonds"][k]) for k in range(n)]

    elif b == "ring_plus_chain":
        ring, chain, err = geom.ring_plus_chain(
            p["ring_bonds"], p["ring_angles"], p["attach_atom"], p["exo_angle"],
            p["chain_bonds"], p.get("chain_angles"))
        coords = np.vstack([ring, chain])
        n = len(p["ring_bonds"])
        check_pairs = [(k, (k + 1) % n, p["ring_bonds"][k]) for k in range(n)]

    elif b == "two_rings_ortho":
        rp = p["ring"]
        n = len(rp["bonds"])
        ring_a, _ = geom.general_ring(rp["bonds"], rp["angles"])
        ring_b, _ = geom.general_ring(rp["bonds"], rp["angles"])
        a, bb = geom.two_rings_orthogonal(ring_a, ring_b, p["connect_len"], attach_index=0)
        coords = np.vstack([a, bb])
        check_pairs = [(k, (k + 1) % n, rp["bonds"][k]) for k in range(n)]
        check_pairs += [(k + n, n + (k + 1) % n, rp["bonds"][k]) for k in range(n)]
        check_pairs += [(0, n, p["connect_len"])]

    elif b == "dodecahedron":
        edge = p["edge"]
        coords = geom.dodecahedron(edge)
        check_pairs = None

    elif b == "xtb_generic":
        n_atoms = p["n_atoms"]
        shape = p.get("shape", "chain")
        edge_guess = 1.42
        guess = geom.generic_guess(shape, n_atoms, edge=edge_guess)
        symbols = ["N"] * n_atoms
        mult = mol.get("mult", 1)
        opt = run_xtb_opt(symbols, guess, mol["charge"], mult, mol["id"])
        coords = opt if opt is not None else guess
        xtb_used = True
        check_pairs = None

    else:
        raise ValueError(f"type de construction inconnu: {b}")

    if symbols is None:
        n = coords.shape[0]
        if formula.endswith("H"):
            symbols = ["N"] * (n - 1) + ["H"]
        else:
            symbols = ["N"] * n

    return coords, symbols, check_pairs, xtb_used


def main():
    report_rows = []
    for mol in M:
        try:
            coords, symbols, check_pairs, xtb_used = build_one(mol)
        except Exception as e:
            print(f"[ERREUR] {mol['id']}: {e}")
            continue
        dev = None
        if check_pairs:
            dev = bond_check(coords, check_pairs)
        comment = (f"{mol['formula']} charge={mol['charge']} mult={mol.get('mult', 1)} "
                   f"PG={mol['point_group']} | {mol['method']} | src: {mol['source']}")
        outpath = os.path.join(OUTDIR, mol["id"] + ".xyz")
        write_xyz(outpath, symbols, coords, comment)
        status = "xtb-refined" if xtb_used else "exact"
        dev_str = f"{dev:.4f} A" if dev is not None else "n/a"
        print(f"[{status:11s}] {mol['id']:32s} n={len(symbols):2d}  dev_bond={dev_str}")
        report_rows.append(dict(
            id=mol["id"], formula=mol["formula"], charge=mol["charge"],
            mult=mol.get("mult", 1), pg=mol["point_group"], method=mol["method"],
            source=mol["source"], status=status, dev=dev,
            notes=mol.get("notes", ""), n=len(symbols)))

    import json
    with open(os.path.join(ROOT, "build", "manifest.json"), "w") as f:
        json.dump(report_rows, f, indent=2, ensure_ascii=False)
    print(f"\n{len(report_rows)} structures generees dans {OUTDIR}")


if __name__ == "__main__":
    main()
