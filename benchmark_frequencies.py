#!/usr/bin/env python3
"""
benchmark_frequencies.py
=========================

Measures wall-clock time for (a) GFN2-xTB geometry optimization and
(b) vibrational frequency calculation (numerical Hessian), across a range
of cluster sizes, to inform whether an energy-window pre-filter is needed
before running frequency verification on every selected candidate.

The Hessian is the expensive part: a numerical Hessian via finite
differences needs ~6N single-point gradient evaluations (2 displacements
x 3 Cartesian directions x N atoms), so its cost is expected to scale
noticeably faster with N than the optimization itself.

Usage
-----
    python benchmark_frequencies.py --sizes 4,6,8,10,12,14,16 -o benchmark_results.csv

Run this on the actual target machine (timings are hardware-dependent);
the size-scaling TREND (time ratio between sizes) is what matters most for
deciding on a filtering strategy, more than the absolute numbers.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from ase import Atoms
from ase.optimize import LBFGS
from ase.vibrations import Vibrations
from tblite.ase import TBLite
from rdkit import Chem
from rdkit.Chem import AllChem

HARTREE_TO_EV = 27.211386245988


def build_prismane_smiles(n_gon: int) -> str:
    """Same systematic [n]prismane builder as cxhx_to_nx.py, reused here to
    get one guaranteed-valid, guaranteed-embeddable, known-size neutral N
    topology per test size without depending on any external file."""
    mol = Chem.RWMol()
    for _ in range(2 * n_gon):
        mol.AddAtom(Chem.Atom(6))
    for i in range(n_gon):
        mol.AddBond(i, (i + 1) % n_gon, Chem.BondType.SINGLE)
        mol.AddBond(n_gon + i, n_gon + (i + 1) % n_gon, Chem.BondType.SINGLE)
        mol.AddBond(i, n_gon + i, Chem.BondType.SINGLE)
    m = mol.GetMol()
    Chem.SanitizeMol(m)
    smi = Chem.MolToSmiles(m)
    # CH -> N isolobal substitution (see cxhx_to_nx.py for the full rationale)
    mol2 = Chem.AddHs(Chem.MolFromSmiles(smi))
    rw = Chem.RWMol(mol2)
    to_remove = []
    for atom in rw.GetAtoms():
        if atom.GetSymbol() == "C":
            h = [nb for nb in atom.GetNeighbors() if nb.GetSymbol() == "H"][0]
            atom.SetAtomicNum(7)
            to_remove.append(h.GetIdx())
    for idx in sorted(to_remove, reverse=True):
        rw.RemoveAtom(idx)
    final = rw.GetMol()
    Chem.SanitizeMol(final)
    return Chem.MolToSmiles(final)


# Test sizes -> SMILES. n_gon=2 gives N4 (degenerate "2-prism" = a 4-ring,
# close enough as a simple, valid, embeddable small test case).
TEST_TOPOLOGIES = {
    4: "N1=NN=N1",
    6: build_prismane_smiles(3),
    8: build_prismane_smiles(4),
    10: build_prismane_smiles(5),
    12: build_prismane_smiles(6),
    14: build_prismane_smiles(7),
    16: build_prismane_smiles(8),
}


def benchmark_one(n_atoms: int, smiles: str, charge: int = 0) -> dict:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.useRandomCoords = True
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        return dict(n_atoms=n_atoms, error="embedding failed")

    positions = mol.GetConformer().GetPositions()
    symbols = [a.GetSymbol() for a in mol.GetAtoms() if a.GetSymbol() != "H"]
    heavy_positions = [positions[i] for i, a in enumerate(mol.GetAtoms()) if a.GetSymbol() != "H"]
    atoms = Atoms(symbols, positions=heavy_positions)
    atoms.calc = TBLite(method="GFN2-xTB", charge=charge, verbosity=0)

    t0 = time.perf_counter()
    opt = LBFGS(atoms, logfile=None)
    opt.run(fmax=0.0005, steps=500)
    t_opt = time.perf_counter() - t0
    n_opt_steps = opt.nsteps

    work_dir = Path(f"/tmp/bench_freq_{n_atoms}")
    t0 = time.perf_counter()
    vib = Vibrations(atoms, name=str(work_dir / "vib"), delta=0.01)
    vib.run()
    freqs = vib.get_frequencies()
    t_freq = time.perf_counter() - t0
    vib.clean()

    n_imag = int((abs(freqs.imag) > 30).sum())

    return dict(
        n_atoms=n_atoms, opt_time_s=t_opt, opt_steps=n_opt_steps,
        freq_time_s=t_freq, n_imaginary=n_imag,
        freq_time_per_dof_ms=(t_freq / (3 * n_atoms)) * 1000,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=str, default="4,6,8,10,12,14,16",
                         help="Comma-separated list of cluster sizes to test "
                              "(must have a built-in test topology, see TEST_TOPOLOGIES)")
    parser.add_argument("-o", "--output", type=str, default="benchmark_results.csv")
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    rows = []
    for n in sizes:
        if n not in TEST_TOPOLOGIES:
            print(f"[SKIP] no built-in test topology for N={n}")
            continue
        print(f"Benchmarking N{n}...")
        result = benchmark_one(n, TEST_TOPOLOGIES[n])
        if "error" in result:
            print(f"  ERROR: {result['error']}")
            continue
        rows.append(result)
        print(f"  optimisation: {result['opt_time_s']:.2f}s ({result['opt_steps']} pas)  |  "
              f"frequences: {result['freq_time_s']:.2f}s  |  "
              f"{result['n_imaginary']} mode(s) imaginaire(s)")

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(f"\nResultats ecrits dans {args.output}")
    print(df.to_string(index=False))

    if len(df) >= 2:
        ratio = df["freq_time_s"].iloc[-1] / df["freq_time_s"].iloc[0]
        n_ratio = df["n_atoms"].iloc[-1] / df["n_atoms"].iloc[0]
        print(f"\nEchelle observee : le calcul de frequences prend x{ratio:.1f} plus longtemps "
              f"quand la taille passe de N{df['n_atoms'].iloc[0]} a N{df['n_atoms'].iloc[-1]} "
              f"(x{n_ratio:.1f} en nombre d'atomes) -- exposant apparent ~ "
              f"{__import__('math').log(ratio)/__import__('math').log(n_ratio):.2f}")


if __name__ == "__main__":
    main()
