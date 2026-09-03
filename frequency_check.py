#!/usr/bin/env python3
"""
frequency_check.py
===================

Post-optimization verification: for a structure already relaxed by
GFN2-xTB (gradient ~0), compute the vibrational Hessian and check whether
it is a genuine local minimum (all vibrational frequencies real/positive)
or a saddle point (one or more imaginary frequencies) -- a symmetric
gradient-zero structure can still be a saddle point along a
symmetry-breaking distortion (Jahn-Teller-like instability), which a
gradient-only optimization can never detect on its own.

If imaginary modes are found, automatically follows the largest one
(small displacement along its eigenvector, then re-optimize) and repeats
until either a genuine minimum is reached or a maximum number of hops is
exhausted -- the standard practical response to discovering a saddle
point via frequency analysis.

Motivating example (see project history): the regular tetrahedral N4
structure (Td symmetry, all N-N distances equal, matching both MAYGEN's
combinatorial output and the tetrahedrane CH->N isolobal substitution) has
a gradient of exactly zero by symmetry, but its Hessian has a genuine
imaginary mode; following it and re-optimizing finds a distorted structure
~8.9 kcal/mol lower in energy (two short N-N ~1.25 A, two medium ~1.48 A,
two long/near-nonbonding ~1.94 A) -- confirming this check is not a
formality, it can change which structure is actually the reported minimum.

Usage
-----
    python frequency_check.py --xyz structure.xyz --charge 0 --gfn 2 \\
        --output-dir ./freq_results

As a library:
    from frequency_check import verify_and_relax_to_minimum
    result = verify_and_relax_to_minimum(atoms, charge=0, gfn=2)
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.optimize import LBFGS
from ase.vibrations import Vibrations
from tblite.ase import TBLite

HARTREE_TO_EV = 27.211386245988
EV_TO_KCALMOL = 23.060548

# Frequencies below this magnitude (cm^-1) are treated as numerical noise
# on the 6 (or 5, linear) near-zero translational/rotational modes, not
# genuine instabilities.
IMAGINARY_NOISE_THRESHOLD_CM1 = 30.0


def _apply_charge(atoms: Atoms, charge: int) -> None:
    """Write the net charge onto the atoms via set_initial_charges (sum =
    net charge). Some tblite versions silently ignore the calculator's
    charge= kwarg, so this is the authoritative way to charge the system."""
    if charge != 0:
        init = [0.0] * len(atoms)
        init[0] = float(charge)
        atoms.set_initial_charges(init)


def _make_calc(charge: int, uhf: int, gfn: int):
    multiplicity = uhf + 1 if uhf else 1
    return TBLite(method=f"GFN{gfn}-xTB", charge=charge, multiplicity=multiplicity, verbosity=0)


def compute_frequencies(atoms: Atoms, work_dir: Path, delta: float = 0.01):
    """Compute vibrational frequencies (cm^-1, complex: imaginary part
    nonzero <=> unstable mode) via a numerical Hessian. Returns
    (frequencies, vib_object) -- the caller is responsible for vib.clean()."""
    work_dir.mkdir(parents=True, exist_ok=True)
    vib = Vibrations(atoms, name=str(work_dir / "vib"), delta=delta)
    vib.run()
    freqs = vib.get_frequencies()
    return freqs, vib


def count_significant_imaginary(freqs: np.ndarray, threshold_cm1: float = IMAGINARY_NOISE_THRESHOLD_CM1) -> int:
    return int(np.sum(np.abs(freqs.imag) > threshold_cm1))


def verify_and_relax_to_minimum(atoms: Atoms, charge: int = 0, uhf: int = 0, gfn: int = 2,
                                 work_dir: Path = None, max_hops: int = 5,
                                 fmax: float = 0.0005, step_size: float = 0.3) -> dict:
    """Ensure `atoms` (already gradient-optimized) sits at a genuine local
    minimum by checking its Hessian and, if needed, following imaginary
    modes downhill with re-optimization. Returns a dict describing the
    outcome: final energy, number of hops needed, whether a true minimum
    was reached, and the final frequencies."""
    work_dir = work_dir or Path("./freq_work")
    atoms = atoms.copy()
    _apply_charge(atoms, charge)
    atoms.calc = _make_calc(charge, uhf, gfn)

    history = []
    for hop in range(max_hops + 1):
        hop_dir = work_dir / f"hop{hop}"
        freqs, vib = compute_frequencies(atoms, hop_dir)
        n_imag = count_significant_imaginary(freqs)
        energy_ev = atoms.get_potential_energy()
        history.append(dict(hop=hop, energy_ev=energy_ev, n_imaginary=n_imag))

        if n_imag == 0:
            vib.clean()
            return dict(is_minimum=True, hops_needed=hop, final_energy_ev=energy_ev,
                        frequencies_cm1=freqs, atoms=atoms, history=history)

        if hop == max_hops:
            vib.clean()
            return dict(is_minimum=False, hops_needed=hop, final_energy_ev=energy_ev,
                        frequencies_cm1=freqs, atoms=atoms, history=history,
                        note=f"Still {n_imag} imaginary mode(s) after {max_hops} hops -- "
                             f"give up, flag as unresolved saddle point.")

        # Follow the largest-magnitude imaginary mode downhill, then reoptimize.
        imag_idx = np.argmax(np.abs(freqs.imag))
        mode = vib.get_mode(imag_idx)
        mode = mode / np.linalg.norm(mode)
        vib.clean()

        atoms.positions += step_size * mode
        _apply_charge(atoms, charge)
        atoms.calc = _make_calc(charge, uhf, gfn)
        opt = LBFGS(atoms, logfile=None)
        opt.run(fmax=fmax, steps=500)

    # unreachable, but keep linters happy
    return dict(is_minimum=False, hops_needed=max_hops, final_energy_ev=atoms.get_potential_energy(),
                frequencies_cm1=freqs, atoms=atoms, history=history)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--xyz", required=True, help="Optimized structure (.xyz)")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--uhf", type=int, default=0)
    parser.add_argument("--gfn", type=int, default=2)
    parser.add_argument("--max-hops", type=int, default=5,
                         help="Max number of imaginary-mode-following attempts (default: 5)")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    atoms = read(args.xyz)
    result = verify_and_relax_to_minimum(
        atoms, charge=args.charge, uhf=args.uhf, gfn=args.gfn,
        work_dir=out_dir / "work", max_hops=args.max_hops,
    )

    print(f"Is true minimum: {result['is_minimum']}")
    print(f"Hops needed: {result['hops_needed']}")
    print(f"Final energy: {result['final_energy_ev']:.6f} eV")
    for h in result["history"]:
        print(f"  hop {h['hop']}: E={h['energy_ev']:.6f} eV, "
              f"{h['n_imaginary']} significant imaginary mode(s)")
    if not result["is_minimum"]:
        print(f"WARNING: {result.get('note', 'not fully resolved')}")

    out_dir.mkdir(parents=True, exist_ok=True)
    write(out_dir / "final_structure.xyz", result["atoms"])
    print(f"Final structure written to {out_dir / 'final_structure.xyz'}")


if __name__ == "__main__":
    main()
