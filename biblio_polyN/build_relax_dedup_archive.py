#!/usr/bin/env python3
"""
build_relax_dedup_archive.py
==============================

Phase C of the archive-extraction pipeline (2026 Archive.zip / 38 new
papers). Builds initial 3D geometries for the 52 "tractable" new species
(exact/bond_params, non-TS, n_atoms<=20) identified in
tractable_new_species.json, relaxes them with GFN2-xTB (tblite), verifies
frequencies, deduplicates against the existing 528-structure pool
(biblio 70 + N_csp 408 + polyN_study 50) and among themselves, and writes
a manifest CSV.

Geometry construction reuses biblio_polyN/build/geom.py's shape
primitives when the bond_params cleanly match one; falls back to
geom.generic_guess(shape, n_atoms) (crude 3D scatter / ring / chain
guess) for cage/ambiguous topologies, since GFN2-xTB relaxation is
expected to refine any reasonable starting point (Niveau 2 convention,
same as elsewhere in this project) -- entries built this way are flagged
LOW_CONFIDENCE_BUILD in the output for manual review.
"""
from __future__ import annotations

import csv
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import networkx as nx

ROOT = Path("/Users/akapulco/polyN")
BIBLIO = ROOT / "biblio_polyN"
sys.path.insert(0, str(BIBLIO / "build"))
sys.path.insert(0, str(ROOT))
import geom  # noqa: E402
from ase import Atoms  # noqa: E402
from ase.io import read, write  # noqa: E402
from ase.optimize import LBFGS  # noqa: E402
from tblite.ase import TBLite  # noqa: E402
from frequency_check import verify_and_relax_to_minimum  # noqa: E402

OUT_XYZ = BIBLIO / "archive_new_xyz"
OUT_XYZ.mkdir(exist_ok=True)
BOND_THRESHOLD = 1.9
HARTREE_TO_EV = 27.211386245988


# --------------------------------------------------------------------
# tblite spin/charge fix (see frequency_check.py _apply_spin / _apply_charge)
# --------------------------------------------------------------------
def _apply_charge(atoms, charge):
    if charge != 0:
        init = [0.0] * len(atoms)
        init[0] = float(charge)
        atoms.set_initial_charges(init)


def _apply_spin(atoms, uhf):
    if uhf != 0:
        init = [0.0] * len(atoms)
        init[0] = float(uhf)
        atoms.set_initial_magnetic_moments(init)


def relax_gfn2(coords, charge, uhf, gfn=2, fmax=0.0005, max_steps=500):
    atoms = Atoms(symbols=["N"] * len(coords), positions=coords)
    _apply_charge(atoms, charge)
    _apply_spin(atoms, uhf)
    mult = uhf + 1 if uhf else 1
    try:
        atoms.calc = TBLite(method=f"GFN{gfn}-xTB", charge=charge, multiplicity=mult, verbosity=0)
        opt = LBFGS(atoms, logfile=None)
        opt.run(fmax=fmax, steps=max_steps)
        e = atoms.get_potential_energy() / HARTREE_TO_EV
        return True, e, atoms
    except Exception as exc:
        return False, str(exc), atoms


# --------------------------------------------------------------------
# geometry construction dispatch
# --------------------------------------------------------------------
def build_geometry(sp: dict):
    """Returns (coords ndarray Nx3, confidence: 'HIGH'|'LOW', build_used str)."""
    n_atoms = int("".join(c for c in sp["formula"] if c.isdigit()) or "0")
    bp = sp.get("bond_params") or {}
    sb = (sp.get("suggested_build") or "").lower()

    def _f(*keys, default=None):
        for k in keys:
            if k in bp and bp[k] is not None:
                v = bp[k]
                if isinstance(v, list):
                    return v
                return v
        return default

    try:
        if "linear" in sb:
            bonds = bp.get("chain_bonds_angstrom") or [bp.get("R_NN_angstrom") or bp.get("edge_A")] * (n_atoms - 1)
            if bonds and len(bonds) == n_atoms - 1:
                coords = np.zeros((n_atoms, 3))
                x = 0.0
                for k, b in enumerate(bonds):
                    x += float(b)
                    coords[k + 1] = [x, 0, 0]
                return coords, "HIGH", "linear"

        if sb == "tetrahedron" or ("tetrahedron" in sb and n_atoms == 4):
            edge = bp.get("edge_angstrom") or bp.get("edge_A")
            if edge:
                return geom.tetrahedron(float(edge)), "HIGH", "tetrahedron"

        if sb == "cube" or ("cube" in sb and n_atoms == 8):
            edge = bp.get("edge_A") or (bp.get("R_ss_angstrom", {}) or {}).get("MP2") if isinstance(bp.get("R_ss_angstrom"), dict) else bp.get("R_ss_angstrom")
            if edge is None:
                edge = 1.52
            return geom.cube(float(edge)), "HIGH", "cube"

        if sb == "dodecahedron":
            edge = bp.get("edge_B3LYP") or bp.get("edge_MP2") or bp.get("edge_HF") or 1.49
            return geom.dodecahedron(float(edge)), "HIGH", "dodecahedron"

        if sb == "regular_ngon":
            edge = bp.get("ring_bond_angstrom") or bp.get("bond_A") or bp.get("edge_angstrom")
            if edge:
                return geom.regular_ngon(n_atoms, float(edge)), "HIGH", "regular_ngon"

        if sb == "rectangle":
            a = bp.get("bond_single_A"); b = bp.get("bond_double_A")
            if a and b:
                return geom.rectangle(float(a), float(b)), "HIGH", "rectangle"

        if "puckered_4ring" in sb:
            edge = bp.get("R12_angstrom"); ang = bp.get("angle_R123_deg")
            if edge and ang:
                return geom.puckered_4ring_d2d(float(edge), float(ang)), "HIGH", "puckered_4ring_d2d"

        if "chain2d" in sb or "chain" in sb and n_atoms <= 12:
            bonds = bp.get("bonds_angstrom") or bp.get("chain_bonds_angstrom")
            angs = bp.get("angles_deg")
            if isinstance(bp.get("bonds_A"), dict):
                bonds = list(bp["bonds_A"].values())
            if isinstance(bp.get("angles_deg"), dict):
                angs = list(bp["angles_deg"].values())
            if bonds and len(bonds) == n_atoms - 1:
                if not angs or len(angs) < n_atoms - 2:
                    fill = angs[-1] if angs else 112.0
                    angs = (angs or []) + [fill] * (n_atoms - 2 - len(angs or []))
                angs = [float(a) if abs(float(a)) <= 180 else 360 - float(a) for a in angs[:n_atoms - 2]]
                return geom.chain2d([float(b) for b in bonds], angs), "HIGH" if (bp.get("angles_deg") and len(bp.get("angles_deg") or []) >= n_atoms - 2) else "LOW", "chain2d"

        if "general_ring" in sb:
            bonds = bp.get("bond_single_A"), bp.get("bond_double_A")
            if all(bonds):
                bl = [bonds[0], bonds[1]] * (n_atoms // 2)
                angs = [125.0] * n_atoms
                coords, _ = geom.general_ring(bl[:n_atoms], angs[:n_atoms])
                return coords, "LOW", "general_ring"

        if "fused_bicyclic" in sb:
            edge = 1.35
            rng = bp.get("bond_range_A")
            if rng:
                edge = float(np.mean(rng))
            n1 = n2 = 5 if n_atoms == 8 else n_atoms // 2 + 1
            return geom.fused_bicyclic(n1, n2, edge), "LOW", "fused_bicyclic"

    except Exception:
        pass

    # explicit atom-indexed edge list (e.g. N18 cage) -> graph-based embedding
    edges_key = next((k for k in bp if k.startswith("edges_")), None)
    if edges_key:
        try:
            edge_list = bp[edges_key]
            G = nx.Graph()
            idx_map = {}
            for e in edge_list:
                a, b = e["pair"]
                for atom in (a, b):
                    if atom not in idx_map:
                        idx_map[atom] = len(idx_map)
                G.add_edge(idx_map[a], idx_map[b], length=e["length"])
            n = len(idx_map)
            if n == n_atoms:
                pos3d = nx.spring_layout(G, dim=3, seed=0, weight=None)
                avg_len = float(np.mean([e["length"] for e in edge_list]))
                coords = np.zeros((n, 3))
                for k, p in pos3d.items():
                    coords[k] = p
                # rescale so mean bonded distance ~ avg_len
                dists = [np.linalg.norm(coords[u] - coords[v]) for u, v in G.edges()]
                scale = avg_len / (np.mean(dists) + 1e-9)
                coords *= scale
                return coords, "HIGH", f"graph_embed:{edges_key}"
        except Exception:
            pass

    # fallback: generic_guess
    shape = "cage"
    text = (str(bp) + " " + str(sp.get("notes", ""))).lower()
    if "chain" in text:
        shape = "chain"
    elif ("cage" in text or "nested" in text) and "ring" in text:
        shape = "cage"
    elif "ring" in text and "bicyclic" not in text and "fused" not in text:
        shape = f"ring{n_atoms}" if 3 <= n_atoms <= 9 else "cage"
    elif "bicyclic" in text or "fused" in text or "pentalene" in text:
        shape = "pentalene" if n_atoms == 8 else "branched"
    elif "branch" in text:
        shape = "branched"
    edge = 1.42
    try:
        coords = geom.generic_guess(shape, n_atoms, edge=edge)
        if len(coords) != n_atoms:
            coords = geom.fibonacci_sphere(n_atoms, radius=1.1 * edge * np.sqrt(n_atoms))
        return coords, "LOW", f"generic_guess:{shape}"
    except Exception:
        return geom.fibonacci_sphere(n_atoms, radius=1.1 * edge * np.sqrt(n_atoms)), "LOW", "fibonacci_fallback"


def build_graph(atoms: Atoms, threshold=BOND_THRESHOLD) -> nx.Graph:
    pos = atoms.get_positions()
    n = len(pos)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(pos[i] - pos[j]) < threshold:
                G.add_edge(i, j)
    return G


# --------------------------------------------------------------------
# existing pool loading
# --------------------------------------------------------------------
def load_existing_pool():
    pool = []  # list of (formula, charge, graph, name)
    for f in sorted((BIBLIO / "xyz").glob("*.xyz")):
        name = f.stem
        # charge from name: look for +/- right after the Nxx formula token
        import re
        m = re.match(r"^(N\d+)([+-]?)", name)
        charge = 0
        if m and m.group(2) == "+":
            charge = 1
        elif m and m.group(2) == "-":
            charge = -1
        try:
            atoms = read(str(f))
        except Exception:
            continue
        formula = f"N{len(atoms)}"
        pool.append((formula, charge, build_graph(atoms), f"biblio:{name}"))

    for sub, chg in [("N_csp", 0), ("polyN_study", 0)]:
        d = ROOT / "structures_externes" / sub / "xyz"
        for f in sorted(d.glob("*.xyz")):
            try:
                atoms = read(str(f))
            except Exception:
                continue
            formula = f"N{len(atoms)}"
            pool.append((formula, chg, build_graph(atoms), f"{sub}:{f.stem}"))
    return pool


def find_duplicate(formula, charge, graph, pool):
    for f2, c2, g2, name in pool:
        if f2 == formula and c2 == charge and len(graph) == len(g2):
            if nx.is_isomorphic(graph, g2):
                return name
    return None


# --------------------------------------------------------------------
# worker
# --------------------------------------------------------------------
def _worker(sp):
    sid = sp["id"]
    formula = sp["formula"]
    charge = int(sp["charge"])
    mult = int(sp.get("mult") or 1)
    uhf = mult - 1
    n_atoms = int("".join(c for c in formula if c.isdigit()))

    row = dict(id=sid, source_file=sp.get("_source_file", ""), formula=formula,
               charge=charge, mult=mult, point_group=sp.get("point_group", ""),
               method=sp.get("method", ""), geometry_type=sp.get("geometry_type", ""),
               build_used="", e_xtb_hartree="", e_xtb_eV_per_atom="",
               freq_converged=False, is_true_minimum="", hops_needed="",
               is_duplicate_of_existing="", is_duplicate_within_new_batch="",
               final_status="BUILD_FAILED", build_confidence="")

    try:
        coords, confidence, build_used = build_geometry(sp)
        row["build_used"] = build_used
        row["build_confidence"] = confidence
        if len(coords) != n_atoms:
            row["final_status"] = "BUILD_FAILED"
            return row, None
    except Exception as exc:
        row["notes_build_error"] = str(exc)
        return row, None

    write(str(OUT_XYZ / f"{sid}_initial.xyz"), Atoms(symbols=["N"] * n_atoms, positions=coords))

    ok, res, atoms = relax_gfn2(coords, charge, uhf)
    if not ok:
        row["final_status"] = "RELAX_FAILED"
        return row, None
    e_ha = res
    row["e_xtb_hartree"] = round(e_ha, 6)
    row["e_xtb_eV_per_atom"] = round(e_ha / n_atoms * HARTREE_TO_EV, 5)
    write(str(OUT_XYZ / f"{sid}_xtbopt.xyz"), atoms)

    work_dir = OUT_XYZ / "freq_work" / sid
    try:
        result = verify_and_relax_to_minimum(atoms, charge=charge, uhf=uhf, gfn=2,
                                              work_dir=work_dir, max_hops=5)
    except Exception as exc:
        row["final_status"] = "FREQ_UNRESOLVED_SADDLE"
        row["notes_freq_error"] = str(exc)
        return row, None

    row["freq_converged"] = True
    row["is_true_minimum"] = result["is_minimum"]
    row["hops_needed"] = result["hops_needed"]
    final_atoms = result["atoms"]
    e_final_ha = result["final_energy_ev"] / HARTREE_TO_EV
    row["e_xtb_hartree"] = round(e_final_ha, 6)
    row["e_xtb_eV_per_atom"] = round(e_final_ha / n_atoms * HARTREE_TO_EV, 5)
    write(str(OUT_XYZ / f"{sid}_final.xyz"), final_atoms)

    if not result["is_minimum"]:
        row["final_status"] = "FREQ_UNRESOLVED_SADDLE"
        return row, None

    graph = build_graph(final_atoms)
    row["final_status"] = "NEW_UNIQUE_MINIMUM"
    return row, (formula, charge, graph, sid)


def main():
    species = json.load(open(BIBLIO / "tractable_new_species.json"))
    print(f"{len(species)} especes a construire/relaxer/verifier.")

    n_jobs = max(1, (mp.cpu_count() or 2) - 1)
    t0 = time.time()
    rows = []
    new_graphs = []  # (formula, charge, graph, id)
    with mp.Pool(n_jobs) as pool:
        for i, (row, gtuple) in enumerate(pool.imap_unordered(_worker, species), 1):
            rows.append(row)
            if gtuple:
                new_graphs.append(gtuple)
            if i % 10 == 0 or i == len(species):
                print(f"  [{i}/{len(species)}] elapsed {time.time()-t0:.0f}s")

    print(f"Construction+relaxation+freq termine en {time.time()-t0:.0f}s.")
    print("Chargement du pool existant (biblio+N_csp+polyN_study) pour dedup...")
    existing_pool = load_existing_pool()
    print(f"  pool existant: {len(existing_pool)} structures")

    rows_by_id = {r["id"]: r for r in rows}
    for formula, charge, graph, sid in new_graphs:
        dup = find_duplicate(formula, charge, graph, existing_pool)
        if dup:
            rows_by_id[sid]["is_duplicate_of_existing"] = dup
            rows_by_id[sid]["final_status"] = "DUPLICATE_OF_EXISTING"

    for i, (f1, c1, g1, id1) in enumerate(new_graphs):
        if rows_by_id[id1]["final_status"] == "DUPLICATE_OF_EXISTING":
            continue
        for f2, c2, g2, id2 in new_graphs[i + 1:]:
            if f1 == f2 and c1 == c2 and len(g1) == len(g2) and nx.is_isomorphic(g1, g2):
                if rows_by_id[id2]["final_status"] not in ("DUPLICATE_OF_EXISTING",):
                    rows_by_id[id2]["is_duplicate_within_new_batch"] = id1
                    rows_by_id[id2]["final_status"] = "DUPLICATE_WITHIN_BATCH"

    out_csv = BIBLIO / "archive_new_structures_results.csv"
    fieldnames = ["id", "source_file", "formula", "charge", "mult", "point_group", "method",
                  "geometry_type", "build_used", "build_confidence", "e_xtb_hartree",
                  "e_xtb_eV_per_atom", "freq_converged", "is_true_minimum", "hops_needed",
                  "is_duplicate_of_existing", "is_duplicate_within_new_batch", "final_status"]
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    from collections import Counter
    status_counts = Counter(r["final_status"] for r in rows)
    print("\n--- Bilan final ---")
    for status, n in status_counts.most_common():
        print(f"  {status}: {n}")
    print(f"Resultats: {out_csv}")


if __name__ == "__main__":
    main()
