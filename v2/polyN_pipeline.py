#!/usr/bin/env python3
"""
polyN_pipeline.py
==================

MAYGEN constitutional isomers -> 3D embedding (RDKit) -> GFN2-xTB optimization
-> selection of the lowest-energy structures per stoichiometry (top-N or
energy window) -> optional DFTB+ re-optimization of the selected subset ->
convex hull of formation energy relative to a reference system (default: N2).

Designed for cluster / molecular allotrope screening (e.g. polynitrogen),
as a companion to an existing MAYGEN + AIRSS/MACE-OFF23/DFTB+ workflow.

Usage
-----
    python polyN_pipeline.py --config config.yaml

Dependencies
------------
    pip install rdkit ase pyyaml pandas scipy matplotlib
    xtb   binary must be in $PATH   (https://github.com/grimme-lab/xtb)
    dftb+ binary must be in $PATH, only needed if dftb.enabled = true

Notes
-----
- All candidates here are treated as isolated (non-periodic) molecules/clusters.
  The DFTB+ step therefore never generates a KPointsAndWeights block (that
  requirement only appears for periodic/supercell calculations), which
  sidesteps the DFTB+ 2025 supercell k-point issue entirely.
- The convex-hull construction auto-selects between two modes:
    * "size"        - single-element systems (e.g. pure N clusters of
                       different sizes). x-axis = number of atoms, y-axis =
                       total formation energy relative to the reference
                       (e.g. N2). This identifies stoichiometries that are
                       stable against disproportionation into other cluster
                       sizes present in the study (+ the reference).
    * "composition" - two-element systems. x-axis = composition fraction,
                       y-axis = formation energy per atom, with the two
                       pure elements pinned at (0,0) and (1,0). Ternary+
                       systems are not handled automatically (see the
                       NotImplementedError message for a suggested fallback).
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:  # pragma: no cover
    Chem = None
    AllChem = None

try:
    from ase.io import write as ase_write
    from ase.io import read as ase_read
    from ase.calculators.dftb import Dftb
    ASE_AVAILABLE = True
except ImportError:  # pragma: no cover
    ASE_AVAILABLE = False

try:
    from tblite.ase import TBLite
    TBLITE_AVAILABLE = True
except ImportError:  # pragma: no cover
    TBLITE_AVAILABLE = False

try:
    from frequency_check import verify_and_relax_to_minimum
    FREQUENCY_CHECK_AVAILABLE = True
except ImportError:  # pragma: no cover
    FREQUENCY_CHECK_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("polyN_pipeline")

HARTREE_TO_EV = 27.211386245988
EV_TO_KCALMOL = 23.060548
HARTREE_TO_KCALMOL = HARTREE_TO_EV * EV_TO_KCALMOL


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    formula: str
    smiles: str
    charge: int = 0
    uhf: int = 0
    conf_id: int = 0
    xyz_init: Optional[str] = None
    xyz_prefilter: Optional[str] = None
    xyz_xtb: Optional[str] = None
    xyz_dftb: Optional[str] = None
    e_prefilter_hartree: Optional[float] = None
    e_xtb_hartree: Optional[float] = None
    e_dftb_hartree: Optional[float] = None
    n_atoms: int = 0
    composition: dict = field(default_factory=dict)
    converged_prefilter: bool = False
    converged_xtb: bool = False
    converged_dftb: bool = False
    is_true_minimum: Optional[bool] = None
    n_freq_hops: int = 0
    n_imaginary_freq: Optional[int] = None
    tag: str = ""

    @property
    def best_energy_hartree(self) -> Optional[float]:
        return self.e_dftb_hartree if self.e_dftb_hartree is not None else self.e_xtb_hartree


# ---------------------------------------------------------------------------
# Formula / composition parsing
# ---------------------------------------------------------------------------

def parse_formula(formula: str) -> dict:
    """Parse a formula such as 'BiN3' or 'N4' into {element: count}."""
    tokens = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
    comp: dict = {}
    for el, cnt in tokens:
        if not el:
            continue
        n = int(cnt) if cnt else 1
        comp[el] = comp.get(el, 0) + n
    return comp


# ---------------------------------------------------------------------------
# Reading MAYGEN outputs
# ---------------------------------------------------------------------------

def load_maygen_smiles(path: Path) -> list:
    """Read a MAYGEN -smi output file (one SMILES per line)."""
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(line.split()[0])
    return out


def load_maygen_sdf(path: Path) -> list:
    """Read a MAYGEN -sdf output file and return canonical SMILES via RDKit."""
    if Chem is None:
        raise RuntimeError("RDKit is required to read SDF files (pip install rdkit).")
    out = []
    suppl = Chem.SDMolSupplier(str(path), removeHs=False)
    for mol in suppl:
        if mol is not None:
            out.append(Chem.MolToSmiles(mol))
    return out


def gather_candidates(cfg: dict) -> list:
    """Build the initial list of Candidate objects from the MAYGEN outputs
    described in cfg['systems'] (one formula -> one file, by default
    '<input_dir>/<formula>.<input_format>')."""
    input_dir = Path(cfg["input_dir"])
    fmt = cfg.get("input_format", "smi")
    candidates = []
    for formula, sys_cfg in cfg["systems"].items():
        fname = sys_cfg.get("file", f"{formula}.{fmt}")
        fpath = input_dir / fname
        if not fpath.exists():
            log.warning("No MAYGEN output for %s at %s -- skipping.", formula, fpath)
            continue

        if fmt == "smi":
            smiles_list = load_maygen_smiles(fpath)
        elif fmt == "sdf":
            smiles_list = load_maygen_sdf(fpath)
        else:
            raise ValueError(f"Unsupported input_format: {fmt}")

        charge = int(sys_cfg.get("charge", 0))
        uhf = int(sys_cfg.get("uhf", 0))
        n_conf = int(sys_cfg.get("n_conformers", 1))

        log.info("Formula %-8s: %d isomers read from %s (charge=%+d, uhf=%d, %d conf/isomer)",
                  formula, len(smiles_list), fpath.name, charge, uhf, n_conf)

        for i, smi in enumerate(smiles_list):
            for c in range(n_conf):
                candidates.append(Candidate(
                    formula=formula, smiles=smi, charge=charge, uhf=uhf, conf_id=c,
                    composition=parse_formula(formula),
                    tag=f"{formula}_iso{i:04d}_conf{c}",
                ))
    return candidates


# ---------------------------------------------------------------------------
# 3D embedding (RDKit)
# ---------------------------------------------------------------------------

def embed_3d(cand: Candidate, workdir: Path) -> bool:
    """Generate an initial 3D geometry from the candidate SMILES (ETKDGv3)."""
    if Chem is None:
        raise RuntimeError("RDKit is required for 3D embedding (pip install rdkit).")
    mol = Chem.MolFromSmiles(cand.smiles)
    if mol is None:
        log.warning("RDKit could not parse SMILES '%s' (%s)", cand.smiles, cand.tag)
        return False
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 42 + cand.conf_id
    params.useRandomCoords = True
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        status = AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=1000 + cand.conf_id)
        if status != 0:
            log.warning("3D embedding failed for %s", cand.tag)
            return False

    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        pass  # MMFF may lack parameters for exotic species (e.g. bare N clusters);
              # xtb will relax the geometry properly regardless.

    cand.n_atoms = mol.GetNumAtoms()
    xyz_path = workdir / f"{cand.tag}_init.xyz"
    Chem.MolToXYZFile(mol, str(xyz_path))
    cand.xyz_init = str(xyz_path)
    return True


# ---------------------------------------------------------------------------
# GFN2-xTB optimization
# ---------------------------------------------------------------------------

def _tblite_optimize(xyz_in: str, run_dir: Path, charge: int, uhf: int,
                      gfn: int, opt_level: str, max_steps: int, out_name: str):
    """Core routine: optimize one structure with tblite/ASE. Returns
    (success, energy_hartree, xyz_path) -- does not touch the Candidate
    object, so it can be reused for both the cheap checkpoint pass (coarse,
    loose) and the accurate refinement pass (GFN2, tight)."""
    if not TBLITE_AVAILABLE:
        raise RuntimeError("tblite is required for the xtb step "
                            "(pip install tblite / conda install -c conda-forge tblite-python).")

    run_dir.mkdir(parents=True, exist_ok=True)
    atoms = ase_read(xyz_in)

    method = f"GFN{gfn}-xTB"
    multiplicity = uhf + 1 if uhf else 1
    try:
        calc = TBLite(method=method, charge=charge, multiplicity=multiplicity, verbosity=0)
        atoms.calc = calc
    except Exception as exc:
        log.warning("tblite calculator setup failed (%s): %s", run_dir.name, exc)
        return False, None, None

    fmax_map = {"crude": 0.01, "sloppy": 0.005, "loose": 0.0025, "lax": 0.002,
                "normal": 0.001, "tight": 0.0005, "vtight": 0.0001, "extreme": 0.00005}
    fmax = fmax_map.get(opt_level, 0.0005)

    from ase.optimize import LBFGS
    log_path = run_dir / "opt.log"
    try:
        opt = LBFGS(atoms, logfile=str(log_path))
        opt.run(fmax=fmax, steps=max_steps)
    except Exception as exc:
        log.warning("tblite/ASE optimization failed (%s): %s", run_dir.name, exc)
        return False, None, None

    if not opt.converged():
        log.warning("Optimization did not converge within %d steps (%s)", max_steps, run_dir.name)
        return False, None, None

    try:
        energy_ev = atoms.get_potential_energy()
    except Exception as exc:
        log.warning("Could not retrieve final energy (%s): %s", run_dir.name, exc)
        return False, None, None

    opt_xyz = run_dir / out_name
    ase_write(opt_xyz, atoms)
    return True, energy_ev / HARTREE_TO_EV, str(opt_xyz)


def run_prefilter_opt(cand: Candidate, workdir: Path, prefilter_cfg: dict) -> bool:
    """Cheap checkpoint optimization used purely as a RANKING PROBE to
    pre-screen a large pool of candidates before the accurate (and much more
    expensive) GFN2-xTB refinement -- it is NOT a warm start for the
    refinement stage. Uses the SAME Hamiltonian as the refinement (GFN2-xTB
    by default) with a low step cap, so the interim energy is on the same
    potential energy surface as the final one (unlike using GFN1-xTB as the
    cheap probe, which showed no real per-call speed advantage over GFN2 in
    tblite and risks probing a different PES entirely).

    IMPORTANT: measured with ASE's LBFGS, continuing/restarting the
    optimizer from this checkpoint's PARTIALLY relaxed geometry is more
    expensive than a fresh optimization from the original embedding (loss
    of useful curvature history when a new optimizer object is created
    partway to the minimum) -- so run_xtb_opt() deliberately restarts from
    cand.xyz_init, not from cand.xyz_prefilter. The prefilter's only job is
    to produce a cheap, comparable energy estimate for ranking."""
    run_dir = workdir / cand.tag
    ok, e_ha, xyz = _tblite_optimize(
        cand.xyz_init, run_dir, cand.charge, cand.uhf,
        gfn=prefilter_cfg.get("gfn", 2),
        opt_level=prefilter_cfg.get("opt_level", "loose"),
        max_steps=prefilter_cfg.get("max_steps", 30),
        out_name="prefilter.xyz",
    )
    if not ok:
        return False
    cand.xyz_prefilter = xyz
    cand.e_prefilter_hartree = e_ha
    cand.converged_prefilter = True
    if cand.n_atoms == 0:
        cand.n_atoms = len(ase_read(xyz))
    return True


def run_xtb_opt(cand: Candidate, workdir: Path, xtb_cfg: dict) -> bool:
    """Accurate GFN2-xTB geometry optimization for one candidate (updates
    cand). Always restarts from the original RDKit embedding (cand.xyz_init),
    never from the prefilter's partial geometry -- see run_prefilter_opt()
    for why: restarting LBFGS from a partially-optimized structure was
    measured to cost MORE steps than a fresh start, not fewer.

    Uses the `tblite` Python/ASE binding rather than shelling out to the xtb
    CLI. The xtb command-line optimizer (ANCOPT) contains a known Fortran
    format-string bug in its progress printout (fixed upstream in xtb PR
    #1278, but not yet in most packaged releases/conda-forge builds as of
    this writing) that crashes with a 'Missing comma between descriptors'
    runtime error on many systems. tblite is a separate, actively
    maintained implementation of the same GFN1/GFN2-xTB Hamiltonians and
    does not go through that buggy code path."""
    run_dir = workdir / cand.tag
    ok, e_ha, xyz = _tblite_optimize(
        cand.xyz_init, run_dir, cand.charge, cand.uhf,
        gfn=xtb_cfg.get("gfn", 2),
        opt_level=xtb_cfg.get("opt_level", "tight"),
        max_steps=xtb_cfg.get("max_steps", 500),
        out_name="xtbopt.xyz",
    )
    if not ok:
        return False
    cand.xyz_xtb = xyz
    cand.e_xtb_hartree = e_ha
    cand.converged_xtb = True
    if cand.n_atoms == 0:
        cand.n_atoms = len(ase_read(xyz))
    return True
    cand.n_atoms = len(atoms)
    return True


def _xtb_worker(args):
    cand, workdir, xtb_cfg = args
    return cand if run_xtb_opt(cand, workdir, xtb_cfg) else None


def _prefilter_worker(args):
    cand, workdir, prefilter_cfg = args
    return cand if run_prefilter_opt(cand, workdir, prefilter_cfg) else None


def _run_parallel(worker_fn, tasks: list, n_jobs: int) -> list:
    results = []
    if n_jobs > 1 and len(tasks) > 1:
        with mp.Pool(n_jobs) as pool:
            for res in pool.imap_unordered(worker_fn, tasks):
                if res is not None:
                    results.append(res)
    else:
        for t in tasks:
            res = worker_fn(t)
            if res is not None:
                results.append(res)
    return results


def _scaled_top_n(base_n: int, n_atoms: int, scaling_cfg: dict) -> int:
    """Scale a top_n cutoff with cluster size: 3N-6 vibrational/conformational
    degrees of freedom grow with N, so a fixed cutoff undersamples larger
    clusters relative to smaller ones."""
    if not scaling_cfg.get("enabled", False) or not n_atoms:
        return base_n
    ref_n_atoms = scaling_cfg.get("ref_n_atoms", 4)
    minimum = scaling_cfg.get("minimum", base_n)
    return max(minimum, round(base_n * n_atoms / ref_n_atoms))


def prefilter_shortlist(candidates: list, prefilter_cfg: dict) -> list:
    """Keep, per formula, a broader shortlist based on the cheap prefilter
    energies -- wider than the final selection.top_n, since the checkpoint
    is only meant to weed out the clearly-bad candidates before the
    expensive GFN2-xTB refinement, not to make the final ranking call."""
    base_n = prefilter_cfg.get("keep_top_n", 30)
    scaling_cfg = prefilter_cfg.get("keep_top_n_scaling", {})
    by_formula = defaultdict(list)
    for c in candidates:
        if c.e_prefilter_hartree is not None:
            by_formula[c.formula].append(c)

    shortlisted = []
    for formula, group in by_formula.items():
        group.sort(key=lambda c: c.e_prefilter_hartree)
        n = _scaled_top_n(base_n, group[0].n_atoms, scaling_cfg)
        keep = group[:n]
        log.info("Formula %-8s (n_atoms=%d): prefilter shortlist %d / %d structures",
                  formula, group[0].n_atoms, len(keep), len(group))
        shortlisted.extend(keep)
    return shortlisted


def optimize_all_xtb(candidates: list, workdir: Path, xtb_cfg: dict) -> list:
    """Embed every candidate, optionally run a cheap checkpoint optimization
    pass to shortlist the most promising structures, then refine the
    shortlist (or everything, if the prefilter is disabled) with accurate
    GFN2-xTB. Splitting the work this way is much cheaper when the initial
    candidate pool is large (many isomers x several conformers), since the
    expensive tight optimization only runs on survivors."""
    embed_dir = workdir / "embed"
    embed_dir.mkdir(parents=True, exist_ok=True)
    embedded = [c for c in candidates if embed_3d(c, embed_dir)]
    log.info("3D embedding succeeded for %d / %d candidates.", len(embedded), len(candidates))

    n_jobs = int(xtb_cfg.get("n_jobs", max(1, (os.cpu_count() or 2) - 1)))

    prefilter_cfg = xtb_cfg.get("prefilter", {})
    if prefilter_cfg.get("enabled", False):
        pf_dir = workdir / "prefilter"
        pf_dir.mkdir(parents=True, exist_ok=True)
        pf_tasks = [(c, pf_dir, prefilter_cfg) for c in embedded]
        log.info("Running prefilter optimizations (GFN%s-xTB, opt_level=%s, %d parallel jobs, "
                  "%d structures)...",
                  prefilter_cfg.get("gfn", 2), prefilter_cfg.get("opt_level", "loose"),
                  n_jobs, len(pf_tasks))
        prefiltered = _run_parallel(_prefilter_worker, pf_tasks, n_jobs)
        log.info("Prefilter converged for %d / %d structures.", len(prefiltered), len(pf_tasks))
        to_refine = prefilter_shortlist(prefiltered, prefilter_cfg)
        log.info("Shortlist for GFN2-xTB refinement: %d structures (out of %d prefiltered).",
                  len(to_refine), len(prefiltered))
    else:
        to_refine = embedded

    xtb_dir = workdir / "xtb"
    xtb_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(c, xtb_dir, xtb_cfg) for c in to_refine]
    log.info("Running GFN2-xTB refinement (%d parallel jobs, %d structures)...",
              n_jobs, len(tasks))
    results = _run_parallel(_xtb_worker, tasks, n_jobs)
    log.info("xtb optimization converged for %d / %d structures.", len(results), len(tasks))
    return results


# ---------------------------------------------------------------------------
# Selection: top-N or energy window, per formula
# ---------------------------------------------------------------------------

def select_lowest(candidates: list, selection_cfg: dict) -> list:
    """Keep, for each formula, either the N lowest-energy structures or all
    structures within an energy window (kcal/mol) above the minimum.
    See _scaled_top_n() for the cluster-size scaling behind top_n_scaling."""
    mode = selection_cfg.get("mode", "top_n")
    scaling_cfg = selection_cfg.get("top_n_scaling", {})

    by_formula = defaultdict(list)
    for c in candidates:
        if c.e_xtb_hartree is not None:
            by_formula[c.formula].append(c)

    selected = []
    for formula, group in by_formula.items():
        group.sort(key=lambda c: c.e_xtb_hartree)
        emin = group[0].e_xtb_hartree
        n_atoms = group[0].n_atoms
        if mode == "top_n":
            base_n = int(selection_cfg.get("top_n", 10))
            n = _scaled_top_n(base_n, n_atoms, scaling_cfg)
            keep = group[:n]
        elif mode == "energy_window":
            window_kcal = float(selection_cfg.get("energy_window_kcal", 5.0))
            keep = [c for c in group
                    if (c.e_xtb_hartree - emin) * HARTREE_TO_KCALMOL <= window_kcal]
        else:
            raise ValueError(f"Unknown selection mode: {mode}")
        log.info("Formula %-8s (n_atoms=%d): keeping %d / %d structures (mode=%s%s)",
                  formula, n_atoms, len(keep), len(group), mode,
                  ", scaled" if (mode == "top_n" and scaling_cfg.get("enabled")) else "")
        selected.extend(keep)
    return selected


# ---------------------------------------------------------------------------
# Optional DFTB+ re-optimization (isolated molecules -> no k-points needed)
# ---------------------------------------------------------------------------

def run_dftb_opt(cand: Candidate, workdir: Path, dftb_cfg: dict) -> bool:
    """Re-optimize a candidate's xtb geometry with DFTB+ via ASE.

    The structure is treated as a non-periodic (pbc=False) cluster, so ASE's
    Dftb calculator never writes a KPointsAndWeights block. This deliberately
    avoids the DFTB+ 2025 requirement that only applies to periodic/supercell
    calculations."""
    if not ASE_AVAILABLE:
        raise RuntimeError("ASE is required for the DFTB+ step (pip install ase).")

    run_dir = workdir / cand.tag
    run_dir.mkdir(parents=True, exist_ok=True)
    atoms = ase_read(cand.xyz_xtb)
    atoms.pbc = False

    elements = sorted(set(atoms.get_chemical_symbols()))
    max_ang = dftb_cfg["max_angular_momenta"]
    missing = [el for el in elements if el not in max_ang]
    if missing:
        raise ValueError(f"Missing dftb.max_angular_momenta entry for {missing}.")

    kwargs = dict(
        label="dftb",
        atoms=atoms,
        Hamiltonian_="DFTB",
        Hamiltonian_SCC="Yes",
        Hamiltonian_SCCTolerance=dftb_cfg.get("scc_tolerance", 1e-7),
        Hamiltonian_Charge=cand.charge,
        Hamiltonian_SlaterKosterFiles_Prefix=dftb_cfg["skf_dir"].rstrip("/") + "/",
        Hamiltonian_SlaterKosterFiles_Separator='"-"',
        Hamiltonian_SlaterKosterFiles_Suffix='".skf"',
        Driver_="GeometryOptimization",
        Driver_LatticeOpt="No",
        Driver_MaxSteps=dftb_cfg.get("max_steps", 500),
        Driver_Convergence_GradAMax=dftb_cfg.get("grad_conv", 1e-4),
    )
    for el in elements:
        kwargs[f"Hamiltonian_MaxAngularMomentum_{el}"] = f'"{max_ang[el]}"'
    if cand.uhf:
        kwargs["Hamiltonian_SpinPolarisation_"] = "Colinear"
        kwargs["Hamiltonian_SpinPolarisation_UnpairedElectrons"] = cand.uhf

    calc = Dftb(directory=str(run_dir), **kwargs)
    atoms.calc = calc
    try:
        energy_ev = atoms.get_potential_energy()
    except Exception as exc:
        log.warning("DFTB+ failed for %s: %s", cand.tag, exc)
        return False

    opt_xyz = run_dir / f"{cand.tag}_dftb.xyz"
    ase_write(opt_xyz, atoms)
    cand.xyz_dftb = str(opt_xyz)
    cand.e_dftb_hartree = energy_ev / HARTREE_TO_EV
    cand.converged_dftb = True
    return True


def reoptimize_all_dftb(candidates: list, workdir: Path, dftb_cfg: dict) -> list:
    dftb_dir = workdir / "dftb"
    dftb_dir.mkdir(parents=True, exist_ok=True)
    results = [c for c in candidates if run_dftb_opt(c, dftb_dir, dftb_cfg)]
    log.info("DFTB+ re-optimization converged for %d / %d structures.",
              len(results), len(candidates))
    return results


def run_frequency_verification(cand: Candidate, workdir: Path, freq_cfg: dict) -> bool:
    """Verify that the candidate's best available geometry is a genuine
    local minimum (all vibrational frequencies real) by computing its
    Hessian, and automatically follow any imaginary mode downhill (with
    re-optimization) until a true minimum is reached or max_hops is
    exhausted. CRITICAL: always uses this candidate's OWN charge (and
    uhf), never a default -- a structure verified with the wrong charge
    would compute frequencies for a physically different species."""
    if not FREQUENCY_CHECK_AVAILABLE:
        raise RuntimeError("frequency_check.py (companion module) is required for this step.")

    xyz_path = cand.xyz_dftb or cand.xyz_xtb
    if xyz_path is None:
        return False

    run_dir = workdir / cand.tag
    atoms = ase_read(xyz_path)
    try:
        result = verify_and_relax_to_minimum(
            atoms,
            charge=cand.charge, uhf=cand.uhf,          # <-- per-candidate charge/uhf, never a default
            gfn=freq_cfg.get("gfn", 2),
            work_dir=run_dir,
            max_hops=freq_cfg.get("max_hops", 5),
            fmax=freq_cfg.get("fmax", 0.0005),
            step_size=freq_cfg.get("step_size", 0.3),
        )
    except Exception as exc:
        log.warning("Frequency verification failed for %s (charge=%+d): %s",
                    cand.tag, cand.charge, exc)
        return False

    cand.is_true_minimum = result["is_minimum"]
    cand.n_freq_hops = result["hops_needed"]
    cand.n_imaginary_freq = count_significant_imaginary_wrapper(result["frequencies_cm1"])

    if result["hops_needed"] > 0:
        # The geometry (and energy) changed while following imaginary
        # modes -- persist the corrected structure and energy.
        final_xyz = run_dir / "post_freq_minimum.xyz"
        ase_write(final_xyz, result["atoms"])
        e_ha = result["final_energy_ev"] / HARTREE_TO_EV
        if cand.xyz_dftb:
            cand.xyz_dftb = str(final_xyz)
            cand.e_dftb_hartree = e_ha
        else:
            cand.xyz_xtb = str(final_xyz)
            cand.e_xtb_hartree = e_ha
        log.info("%s: was a saddle point (%d imaginary mode(s) initially), relaxed to true "
                  "minimum after %d hop(s), E changed by %.2f kcal/mol", cand.tag,
                  result["history"][0]["n_imaginary"], result["hops_needed"],
                  (result["history"][0]["energy_ev"] - result["final_energy_ev"]) * EV_TO_KCALMOL)

    if not result["is_minimum"]:
        log.warning("%s: still %s after %d hops -- flagged as unresolved saddle point",
                    cand.tag, result.get("note", "not a minimum"), result["hops_needed"])

    return True


def count_significant_imaginary_wrapper(freqs) -> int:
    import numpy as _np
    return int(_np.sum(_np.abs(freqs.imag) > 30.0))


def _freq_worker(args):
    cand, workdir, freq_cfg = args
    ok = run_frequency_verification(cand, workdir, freq_cfg)
    return cand if ok else None


def verify_all_minima(candidates: list, workdir: Path, cfg: dict) -> list:
    """Run frequency verification on the (already selected and refined)
    candidates. Applied only here -- not on the full raw candidate pool --
    since the Hessian is far more expensive than a single-point/optimization
    (see benchmark_frequencies.py): ~6N extra energy/gradient evaluations
    per structure, scaling roughly as N^1.7-2.2 in practice."""
    freq_cfg = cfg.get("frequency", {})
    if not freq_cfg.get("enabled", True):
        log.info("Frequency verification disabled (frequency.enabled: false).")
        return candidates

    freq_dir = workdir / "frequency"
    freq_dir.mkdir(parents=True, exist_ok=True)
    n_jobs = int(freq_cfg.get("n_jobs", cfg.get("xtb", {}).get("n_jobs", 4)))
    tasks = [(c, freq_dir, freq_cfg) for c in candidates]
    log.info("Running frequency verification (%d parallel jobs, %d structures)...",
              n_jobs, len(tasks))
    results = _run_parallel(_freq_worker, tasks, n_jobs)

    n_saddle_fixed = sum(1 for c in results if c.n_freq_hops > 0 and c.is_true_minimum)
    n_unresolved = sum(1 for c in results if not c.is_true_minimum)
    log.info("Frequency verification: %d / %d confirmed as true minima "
              "(%d were saddle points successfully relaxed, %d unresolved after max_hops).",
              sum(1 for c in results if c.is_true_minimum), len(tasks),
              n_saddle_fixed, n_unresolved)
    return results


# ---------------------------------------------------------------------------
# Reference chemical potentials
# ---------------------------------------------------------------------------

def compute_reference(cfg: dict, workdir: Path, do_dftb: bool = False) -> dict:
    """Optimize the reference system (default N2) and return
    {element: energy_per_atom_hartree}. A pre-computed value can be supplied
    directly via convex_hull.reference_energies_hartree_per_atom to skip
    the calculation."""
    ch_cfg = cfg["convex_hull"]
    if "reference_energies_hartree_per_atom" in ch_cfg:
        return ch_cfg["reference_energies_hartree_per_atom"]

    ref_cfg = ch_cfg["reference"]
    ref_formula = ref_cfg.get("formula", "N2")
    ref_charge = int(ref_cfg.get("charge", 0))
    comp = parse_formula(ref_formula)
    if len(comp) != 1:
        raise ValueError("The default reference must be a single-element system "
                          "(e.g. N2). For multi-element references, supply "
                          "convex_hull.reference_energies_hartree_per_atom directly.")
    element, n_atoms = next(iter(comp.items()))

    smi_map = {"N": "N#N", "P": "P#P", "H": "[H][H]", "O": "O=O",
               "S": "S=S", "F": "F[F]", "Cl": "ClCl"}
    smi = ref_cfg.get("smiles", smi_map.get(element))
    if smi is None:
        raise ValueError(f"No default SMILES known for reference element '{element}'; "
                          f"set convex_hull.reference.smiles in the config.")

    ref_cand = Candidate(formula=ref_formula, smiles=smi, charge=ref_charge,
                          composition=comp, tag=f"REF_{ref_formula}")
    ref_dir = workdir / "reference"
    ref_dir.mkdir(parents=True, exist_ok=True)
    if not embed_3d(ref_cand, ref_dir):
        raise RuntimeError(f"Failed to embed reference structure {ref_formula}")
    if not run_xtb_opt(ref_cand, ref_dir, cfg["xtb"]):
        raise RuntimeError(f"Failed to optimize reference structure {ref_formula} with xtb")
    if do_dftb:
        run_dftb_opt(ref_cand, ref_dir, cfg["dftb"])

    e_ref = ref_cand.best_energy_hartree
    log.info("Reference %s: E = %.6f Ha  (%.6f Ha/atom of %s)",
              ref_formula, e_ref, e_ref / n_atoms, element)
    return {element: e_ref / n_atoms}


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def charge_family(charge: int) -> str:
    """Classify a species by total charge. Convex hulls are built separately
    per family: comparing formation energies across different charge states
    on a single hull is not methodologically sound, for two reasons:
    (1) a real decomposition pathway conserves total charge, so a cation can
    only fragment into a cation + neutrals, an anion into an anion +
    neutrals -- never a cation into an anion or vice versa; and
    (2) semi-empirical/tight-binding methods (GFN2-xTB, DFTB) are not
    guaranteed to place different charge states on a mutually consistent
    absolute energy scale (ionization energies and electron affinities are a
    known weak point of these methods)."""
    if charge < 0:
        return "anion"
    if charge > 0:
        return "cation"
    return "neutral"


def formation_energy(cand: Candidate, mu: dict) -> float:
    e_ref = sum(mu[el] * n for el, n in cand.composition.items())
    return cand.best_energy_hartree - e_ref


def build_results_table(candidates: list, mu: dict) -> pd.DataFrame:
    if not candidates:
        return pd.DataFrame(columns=[
            "formula", "family", "tag", "charge", "n_atoms",
            "e_xtb_hartree", "e_dftb_hartree", "e_formation_hartree",
            "e_formation_ev_per_atom", "e_formation_kcalmol", "xyz",
            "is_true_minimum", "n_freq_hops", "n_imaginary_freq",
        ])
    rows = []
    for c in candidates:
        e_form = formation_energy(c, mu)
        rows.append(dict(
            formula=c.formula, family=charge_family(c.charge), tag=c.tag,
            charge=c.charge, n_atoms=c.n_atoms,
            e_xtb_hartree=c.e_xtb_hartree, e_dftb_hartree=c.e_dftb_hartree,
            e_formation_hartree=e_form,
            e_formation_ev_per_atom=(e_form / c.n_atoms) * HARTREE_TO_EV,
            e_formation_kcalmol=e_form * HARTREE_TO_KCALMOL,
            xyz=c.xyz_dftb or c.xyz_xtb,
            is_true_minimum=c.is_true_minimum,
            n_freq_hops=c.n_freq_hops,
            n_imaginary_freq=c.n_imaginary_freq,
        ))
    return pd.DataFrame(rows).sort_values(["family", "formula", "e_formation_hartree"])


# ---------------------------------------------------------------------------
# Convex hull
# ---------------------------------------------------------------------------

def lower_hull_1d(points: np.ndarray) -> np.ndarray:
    """Indices (into `points`) on the lower convex hull of (x, y) points
    (standard Andrew's monotone-chain lower hull)."""
    order = np.argsort(points[:, 0])
    if len(order) < 2:
        return order
    lower = [order[0]]
    for idx in order[1:]:
        while len(lower) >= 2:
            x1, y1 = points[lower[-2]]
            x2, y2 = points[lower[-1]]
            x3, y3 = points[idx]
            cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
            if cross <= 0:  # not a strict left turn -> the middle point is not on the lower hull
                lower.pop()
            else:
                break
        lower.append(idx)
    return np.array(lower)


def convex_hull_single_element(df: pd.DataFrame) -> pd.DataFrame:
    """Size-based stability hull for ONE charge family: x = number of atoms,
    y = formation energy per atom. A species is "on the hull" if no
    combination of other members of the SAME family (same charge) gives a
    lower per-atom energy for that composition -- i.e. it is not expected to
    disproportionate into other sizes present within that family.

    Note: using per-atom energy (rather than the extensive total formation
    energy) is the convention requested for readability and for comparing
    families with different atom-count spacing (odd vs even N); for a
    fully rigorous grand-canonical disproportionation test the hull should
    strictly be built on total (extensive) formation energy vs n_atoms --
    the two give the same qualitative picture in practice but can differ in
    edge cases. Ask for the extensive-energy variant if you need it."""
    best = df.loc[df.groupby("formula")["e_formation_ev_per_atom"].idxmin()].copy()
    best = best.sort_values("n_atoms").reset_index(drop=True)
    pts = np.column_stack([
        best["n_atoms"].to_numpy(dtype=float),
        best["e_formation_ev_per_atom"].to_numpy(dtype=float),
    ])
    hull_idx = set(lower_hull_1d(pts).tolist())
    best["on_hull"] = [i in hull_idx for i in range(len(best))]
    hull_pts = pts[sorted(hull_idx)]
    best["e_above_hull_ev_per_atom"] = [
        row.e_formation_ev_per_atom - np.interp(row.n_atoms, hull_pts[:, 0], hull_pts[:, 1])
        for row in best.itertuples()
    ]
    return best


def convex_hull_binary(df: pd.DataFrame, elements: list) -> pd.DataFrame:
    """Composition-fraction hull for a two-element system, pure elements
    pinned at (0,0) and (1,0)."""
    best = df.loc[df.groupby("formula")["e_formation_hartree"].idxmin()].copy()

    def frac_b(row):
        comp = parse_formula(row["formula"])
        na, nb = comp.get(elements[0], 0), comp.get(elements[1], 0)
        return nb / (na + nb) if (na + nb) else 0.0

    best["x_frac"] = best.apply(frac_b, axis=1)
    pts = np.column_stack([
        np.concatenate([[0.0, 1.0], best["x_frac"].to_numpy()]),
        np.concatenate([[0.0, 0.0], best["e_formation_ev_per_atom"].to_numpy()]),
    ])
    hull_idx = set(lower_hull_1d(pts).tolist()) - {0, 1}
    best["on_hull"] = [(i + 2) in hull_idx for i in range(len(best))]
    hull_pts = pts[sorted(hull_idx | {0, 1})]
    best["e_above_hull_ev_per_atom"] = [
        row.e_formation_ev_per_atom - np.interp(row.x_frac, hull_pts[:, 0], hull_pts[:, 1])
        for row in best.itertuples()
    ]
    return best


def build_convex_hulls_by_family(df: pd.DataFrame, cfg: dict) -> dict:
    """Build one convex hull PER CHARGE FAMILY (neutral / cation / anion).
    Families are never mixed on the same hull -- see charge_family() for why."""
    elements = sorted({el for comp in df["formula"].map(parse_formula) for el in comp})
    mode = cfg["convex_hull"].get("mode", "auto")
    if mode == "auto":
        mode = "size" if len(elements) == 1 else "composition"

    hulls = {}
    for family, sub in df.groupby("family"):
        if mode == "size":
            hulls[family] = convex_hull_single_element(sub)
        elif mode == "composition":
            if len(elements) != 2:
                raise NotImplementedError(
                    "Automatic composition-hull plotting only supports binary systems. "
                    "For ternary+ systems, use the exported results.csv with pymatgen's "
                    "PhaseDiagram, or ask for this script to be extended.")
            hulls[family] = convex_hull_binary(sub, elements)
        else:
            raise ValueError(f"Unknown convex_hull.mode: {mode}")
    return hulls, mode


def plot_hull(hull_df: pd.DataFrame, mode: str, out_path: Path, ref_formula: str, family: str = None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4.5))

    if mode == "size":
        x, y = hull_df["n_atoms"], hull_df["e_formation_ev_per_atom"]
        xlabel, ylabel = "Number of atoms", f"Formation energy per atom vs {ref_formula} (eV)"
    else:
        x, y = hull_df["x_frac"], hull_df["e_formation_ev_per_atom"]
        xlabel, ylabel = "Composition fraction", "Formation energy per atom (eV)"

    stable = hull_df["on_hull"]
    ax.scatter(x[~stable], y[~stable], color="0.6", label="metastable", zorder=2)
    ax.scatter(x[stable], y[stable], color="crimson", label="on hull", zorder=3)
    order = np.argsort(x[stable].to_numpy())
    ax.plot(x[stable].to_numpy()[order], y[stable].to_numpy()[order],
            color="crimson", lw=1.2, zorder=1)
    for _, row in hull_df.iterrows():
        ax.annotate(row["formula"], (row[x.name], row[y.name]), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if family:
        ax.set_title(f"Famille : {family}")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    log.info("Convex hull plot saved to %s", out_path)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="YAML config file")
    parser.add_argument("--skip-dftb", action="store_true",
                         help="Force-skip the DFTB+ step even if enabled in the config")
    parser.add_argument("--skip-freq", action="store_true",
                         help="Force-skip the frequency verification step even if enabled")
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    out_dir = Path(cfg.get("output_dir", "./results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = out_dir / "work"
    work_dir.mkdir(exist_ok=True)

    log.info("== Step 1/6: reading MAYGEN outputs ==")
    candidates = gather_candidates(cfg)
    log.info("Total candidates: %d", len(candidates))

    log.info("== Step 2/6: GFN2-xTB optimization ==")
    candidates = optimize_all_xtb(candidates, work_dir, cfg["xtb"])

    log.info("== Step 3/6: selecting lowest-energy structures per stoichiometry ==")
    selected = select_lowest(candidates, cfg["selection"])

    do_dftb = cfg.get("dftb", {}).get("enabled", False) and not args.skip_dftb
    if do_dftb:
        log.info("== Step 4/6: DFTB+ re-optimization of the selected subset ==")
        selected = reoptimize_all_dftb(selected, work_dir, cfg["dftb"])
    else:
        log.info("== Step 4/6: skipped (DFTB+ disabled) ==")

    do_freq = cfg.get("frequency", {}).get("enabled", True) and not args.skip_freq
    if do_freq:
        log.info("== Step 5/6: frequency verification (true minima only) ==")
        selected = verify_all_minima(selected, work_dir, cfg)
    else:
        log.info("== Step 5/6: skipped (frequency verification disabled) ==")

    log.info("== Step 6/6: reference & convex hull ==")
    mu = compute_reference(cfg, work_dir, do_dftb=do_dftb)
    df = build_results_table(selected, mu)
    df.to_csv(out_dir / "results.csv", index=False)
    log.info("Full results table written to %s", out_dir / "results.csv")

    hulls, mode = build_convex_hulls_by_family(df, cfg)
    ref_formula = cfg["convex_hull"]["reference"].get("formula", "N2")

    log.info("Done. Stable stoichiometries (on hull), by charge family:")
    for family, hull_df in hulls.items():
        hull_df.to_csv(out_dir / f"convex_hull_{family}.csv", index=False)
        plot_hull(hull_df, mode, out_dir / f"convex_hull_{family}.png", ref_formula, family=family)
        log.info("  [%s]", family)
        for _, row in hull_df[hull_df["on_hull"]].iterrows():
            log.info("    %s", row["formula"])


if __name__ == "__main__":
    main()
