#!/usr/bin/env python3
"""
polyN_pipeline.py
==================

Topology generation (isolobal harvest [default] / random / MAYGEN) -> 3D
embedding (RDKit) -> GFN2-xTB optimization (checkpoint + refinement) ->
selection of the lowest-energy structures per stoichiometry -> frequency
verification (confirm true minima; follow imaginary modes downhill) ->
convex hulls of formation energy, built separately per charge family.

Designed for cluster / molecular allotrope screening (e.g. polynitrogen),
suitable for building a clean, minimum-verified dataset (e.g. to train a
generative model).

Usage
-----
    python polyN_pipeline.py --config config.yaml                    # isolobal (default)
    python polyN_pipeline.py --config config.yaml --generators isolobal random
    python polyN_pipeline.py --config config.yaml --generators none  # pre-existing inputs

Dependencies
------------
    pip install rdkit ase pyyaml pandas scipy matplotlib pubchempy
    xtb via tblite (Python) -- no external binary needed for optimization.

Energy references (per charge family)
-------------------------------------
- neutral family: formation energy relative to N2 -- a genuine physical
  quantity (energy to assemble an even-N neutral allotrope from N2 gas).
- cation / anion families: relative to the most stable member of that same
  charged family (lowest energy per atom, across all sizes). This is a
  per-family RELATIVE zero for the convex-hull Y-axis, NOT a physical
  formation enthalpy -- comparing absolute energies across charge states is
  not sound with tight-binding methods, so each charged family is referenced
  only to itself.

Notes
-----
- All candidates are treated as isolated (non-periodic) molecules/clusters.
- The convex-hull construction auto-selects between "size" (single-element,
  e.g. pure N clusters of different sizes: x = number of atoms) and
  "composition" (two-element) modes.
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
    e_prefilter_hartree: Optional[float] = None
    e_xtb_hartree: Optional[float] = None
    n_atoms: int = 0
    composition: dict = field(default_factory=dict)
    converged_prefilter: bool = False
    converged_xtb: bool = False
    is_true_minimum: Optional[bool] = None
    n_freq_hops: int = 0
    n_imaginary_freq: Optional[int] = None
    tag: str = ""

    @property
    def best_energy_hartree(self) -> Optional[float]:
        return self.e_xtb_hartree


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


def run_generators(cfg: dict, generators: list, work_dir: Path) -> None:
    """Produce topology .smi files into cfg['input_dir'] using the selected
    generator(s). Multiple generators can be combined -- their outputs for
    the same (size, family) accumulate, giving a richer, deduplicated pool.

    Available generators (point 2):
      - 'isolobal' (2c, DEFAULT): PubChem hydrocarbon harvest + isolobal
        C->N substitution (harvest_hydrocarbons.py + cxhx_to_nx.py). The
        most chemically-motivated source: every topology derives from a
        real, experimentally-catalogued hydrocarbon.
      - 'maygen'   (2a): exhaustive constitutional-isomer enumeration via an
        external MAYGEN jar (must be available; path from cfg or --maygen-jar).
      - 'random'   (2b): random valence-correct molecular-graph sampling
        (random_structure_generator.py), for any size/charge.

    This function only PREPARES input files; the rest of the pipeline then
    reads them via gather_candidates() exactly as before, so pre-existing
    input files (e.g. a manually curated maygen_output/) still work with
    the default of not running any generator."""
    if not generators:
        return
    input_dir = Path(cfg["input_dir"])
    input_dir.mkdir(parents=True, exist_ok=True)
    log.info("Running topology generator(s): %s", ", ".join(generators))

    for gen in generators:
        if gen == "isolobal":
            _run_isolobal_generator(cfg, input_dir, work_dir)
        elif gen == "random":
            _run_random_generator(cfg, input_dir)
        elif gen == "maygen":
            _run_maygen_generator(cfg, input_dir)
        elif gen == "geng":
            _run_geng_generator(cfg, input_dir)
        else:
            raise ValueError(f"Unknown generator '{gen}' "
                              f"(choose from: isolobal, random, maygen, geng).")


def _run_isolobal_generator(cfg: dict, input_dir: Path, work_dir: Path) -> None:
    """Harvest hydrocarbons from PubChem, then isolobally substitute C->N.
    Delegates to the companion scripts via subprocess so their (already
    tested) logic is reused verbatim rather than duplicated."""
    import subprocess
    gen_cfg = cfg.get("generators", {}).get("isolobal", {})
    min_c = int(gen_cfg.get("min_c", 4))
    max_c = int(gen_cfg.get("max_c", 16))
    max_abs_charge = int(gen_cfg.get("max_abs_charge", 1))
    hc_file = work_dir / "hydrocarbons_harvested.smi"

    here = Path(__file__).parent
    if not hc_file.exists() or gen_cfg.get("force_reharvest", False):
        log.info("  [isolobal] harvesting hydrocarbons C%d-C%d from PubChem...", min_c, max_c)
        subprocess.run(
            ["python3", str(here / "harvest_hydrocarbons.py"),
             "--min-c", str(min_c), "--max-c", str(max_c), "-o", str(hc_file), "--resume"],
            check=True)
    else:
        log.info("  [isolobal] reusing existing harvest at %s (set force_reharvest to redo)", hc_file)

    log.info("  [isolobal] isolobal C->N substitution (|charge| <= %d)...", max_abs_charge)
    subprocess.run(
        ["python3", str(here / "cxhx_to_nx.py"),
         "--smiles-file", str(hc_file), "--max-abs-charge", str(max_abs_charge),
         "-o", str(input_dir)],
        check=True)


def _run_random_generator(cfg: dict, input_dir: Path) -> None:
    """Generate random valence-correct topologies for each system in the
    config, using random_structure_generator.py."""
    import subprocess
    gen_cfg = cfg.get("generators", {}).get("random", {})
    k = int(gen_cfg.get("k_per_system", 200))
    seed = gen_cfg.get("seed", 42)
    here = Path(__file__).parent

    for formula, sys_cfg in cfg["systems"].items():
        comp = parse_formula(formula)
        if len(comp) != 1 or "N" not in comp:
            continue
        n = comp["N"]
        charge = int(sys_cfg.get("charge", 0))
        out_file = input_dir / sys_cfg.get("file", f"{formula}.smi")
        cmd = ["python3", str(here / "random_structure_generator.py"),
               "-n", str(n), "-k", str(k), "--seed", str(seed), "-o", str(out_file)]
        if charge != 0:
            cmd += ["--target-charge", str(charge)]
        log.info("  [random] %s (n=%d, charge=%+d, k=%d)", formula, n, charge, k)
        subprocess.run(cmd, check=True)


def _run_maygen_generator(cfg: dict, input_dir: Path) -> None:
    """Enumerate constitutional isomers with an external MAYGEN jar (neutral
    families only -- MAYGEN's charge handling is unreliable for this use,
    see the manual)."""
    import subprocess
    gen_cfg = cfg.get("generators", {}).get("maygen", {})
    jar = gen_cfg.get("jar")
    if not jar or not Path(jar).exists():
        raise FileNotFoundError(
            "MAYGEN generator selected but no valid jar found. Set "
            "generators.maygen.jar in the config to the MAYGEN .jar path.")
    for formula, sys_cfg in cfg["systems"].items():
        comp = parse_formula(formula)
        if len(comp) != 1 or "N" not in comp:
            continue
        if int(sys_cfg.get("charge", 0)) != 0:
            log.info("  [maygen] skipping charged system %s (neutral only)", formula)
            continue
        out_file = input_dir / sys_cfg.get("file", f"{formula}.smi")
        log.info("  [maygen] enumerating %s ...", formula)
        subprocess.run(
            ["java", "-jar", jar, "-f", formula, "-smi", "-o", str(out_file)],
            check=True)


def _run_geng_generator(cfg: dict, input_dir: Path) -> None:
    """Exhaustively enumerate topologies with geng (nauty) + bond-order
    assignment, via geng_enumerate.py. Formal charges are assigned at the
    graph stage (before RDKit sanitization), so the neutral, cation, and
    anion families all emerge from the same connected-graph set without any
    phantom-hydrogen saturation. Runs once per distinct atom count present in
    the config's systems."""
    import subprocess
    gen_cfg = cfg.get("generators", {}).get("geng", {})
    max_abs_charge = int(gen_cfg.get("max_abs_charge", 1))
    max_graphs = gen_cfg.get("max_graphs")       # None = exhaustive
    seed = int(gen_cfg.get("seed", 42))
    here = Path(__file__).parent

    # Collect the distinct atom counts N requested by the config.
    n_values = set()
    for formula in cfg["systems"]:
        comp = parse_formula(formula)
        if len(comp) == 1 and "N" in comp:
            n_values.add(comp["N"])

    for n in sorted(n_values):
        cmd = ["python3", str(here / "geng_enumerate.py"),
               "-n", str(n), "--max-abs-charge", str(max_abs_charge),
               "--seed", str(seed), "-o", str(input_dir)]
        if max_graphs is not None:
            cmd += ["--max-graphs", str(int(max_graphs))]
        log.info("  [geng] exhaustive enumeration for N=%d ...", n)
        subprocess.run(cmd, check=True)


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
        # IMPORTANT: some tblite versions silently IGNORE the constructor's
        # charge= kwarg (verified: N5+ and N5- then optimize to identical
        # energies). The mechanism that reliably applies the total charge is
        # ASE's set_initial_charges, whose sum tblite reads as the net charge.
        # We set BOTH: the per-atom charges (authoritative for tblite here)
        # and the kwarg (harmless, and correct for versions that honor it).
        if charge != 0:
            init_charges = [0.0] * len(atoms)
            init_charges[0] = float(charge)   # sum = net charge
            atoms.set_initial_charges(init_charges)
        # Same issue for spin: tblite.ase.TBLite reads the spin state ONLY
        # from atoms.get_initial_magnetic_moments().sum(), never from the
        # multiplicity= kwarg below (verified: multiplicity=1 vs 3 on the
        # same charge-0 geometry with no magnetic moment set give IDENTICAL
        # energy). Without this, any candidate requesting uhf!=0 silently
        # runs at whatever spin tblite defaults to instead.
        if uhf != 0:
            init_moments = [0.0] * len(atoms)
            init_moments[0] = float(uhf)
            atoms.set_initial_magnetic_moments(init_moments)
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

    xyz_path = cand.xyz_xtb
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
        cand.xyz_xtb = str(final_xyz)
        cand.e_xtb_hartree = result["final_energy_ev"] / HARTREE_TO_EV
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

def compute_neutral_reference(cfg: dict, workdir: Path) -> dict:
    """Optimize the neutral reference system (default N2) at GFN2-xTB and
    return {element: energy_per_atom_hartree}.

    This reference applies ONLY to the neutral family (point 3a): for an
    even-N neutral polynitrogen NxN, the formation energy relative to
    (x/2) N2 is a genuine physical quantity (energy to assemble the
    allotrope from ordinary dinitrogen gas).

    The charged families (cation/anion) do NOT use this reference -- see
    charged_family_reference(), which derives a per-family zero from the
    most stable member of that same charge family (point 3b)."""
    ch_cfg = cfg["convex_hull"]
    if "reference_energies_hartree_per_atom" in ch_cfg:
        return ch_cfg["reference_energies_hartree_per_atom"]

    ref_cfg = ch_cfg.get("reference", {})
    ref_formula = ref_cfg.get("formula", "N2")
    comp = parse_formula(ref_formula)
    if len(comp) != 1:
        raise ValueError("The neutral reference must be a single-element system "
                          "(e.g. N2). For multi-element references, supply "
                          "convex_hull.reference_energies_hartree_per_atom directly.")
    element, n_atoms = next(iter(comp.items()))

    smi_map = {"N": "N#N", "P": "P#P", "H": "[H][H]", "O": "O=O",
               "S": "S=S", "F": "F[F]", "Cl": "ClCl"}
    smi = ref_cfg.get("smiles", smi_map.get(element))
    if smi is None:
        raise ValueError(f"No default SMILES known for reference element '{element}'; "
                          f"set convex_hull.reference.smiles in the config.")

    ref_cand = Candidate(formula=ref_formula, smiles=smi, charge=0,
                          composition=comp, tag=f"REF_{ref_formula}")
    ref_dir = workdir / "reference"
    ref_dir.mkdir(parents=True, exist_ok=True)
    if not embed_3d(ref_cand, ref_dir):
        raise RuntimeError(f"Failed to embed reference structure {ref_formula}")
    if not run_xtb_opt(ref_cand, ref_dir, cfg["xtb"]):
        raise RuntimeError(f"Failed to optimize reference structure {ref_formula} with xtb")

    e_ref = ref_cand.best_energy_hartree
    log.info("Neutral reference %s: E = %.6f Ha  (%.6f Ha/atom of %s)",
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


def neutral_formation_energy(cand: Candidate, mu: dict) -> float:
    """Formation energy of a NEUTRAL species relative to the elemental
    reference N2: E(Nx) - (x/2) E(N2). A genuine physical quantity."""
    e_ref = sum(mu[el] * n for el, n in cand.composition.items())
    return cand.best_energy_hartree - e_ref


def charged_reaction_energy(cand: Candidate, e_n2: float, e_ref5: float) -> float:
    """Reaction energy for a monocharged species relative to N2 + the
    experimental N5-charge reference, via the balanced reaction

        (Nx)^q  ->  ((x-5)/2) N2  +  (N5)^q          (q = +1 or -1)

    i.e.  E_reaction = E(Nx^q) - [ (x-5)/2 * E(N2) + E(N5^q) ].

    Nitrogen balance requires x odd and x >= 5 (so (x-5)/2 is a
    non-negative integer). N5 itself gives 0 by construction (it IS the
    reference). This is a physically meaningful decomposition energy: how
    much the cluster is stabilized/destabilized relative to breaking down
    into N2 gas plus the reference pentazolate/pentazolium ion."""
    x = cand.n_atoms
    coeff_n2 = (x - 5) / 2.0
    return cand.best_energy_hartree - (coeff_n2 * e_n2 + e_ref5)


def build_results_table(candidates: list, mu: dict, charged_refs: dict) -> pd.DataFrame:
    """Build the results table with per-family reference energies.

    - neutral: E(Nx) - (x/2) E(N2)          [reference: N2]
    - cation:  E(Nx+) - ((x-5)/2 E(N2) + E(N5+))   [reference: N2 + N5+]
    - anion:   E(Nx-) - ((x-5)/2 E(N2) + E(N5-))   [reference: N2 + N5-]

    charged_refs must provide 'e_n2', and 'e_n5_cation' / 'e_n5_anion'
    (Hartree) for whichever charged families are present. These come from
    the config (mandatory pre-computed values) or, if N5+/N5- is itself in
    the run, from the lowest-energy N5 of that family."""
    columns = [
        "rank", "formula", "family", "tag", "charge", "n_atoms",
        "e_xtb_hartree", "e_per_atom_hartree",
        "e_reaction_hartree", "e_reaction_ev_per_atom", "e_reaction_kcalmol",
        "reference_reaction", "smiles", "xyz",
        "is_true_minimum", "n_freq_hops", "n_imaginary_freq",
    ]
    if not candidates:
        return pd.DataFrame(columns=columns)

    e_n2 = charged_refs.get("e_n2")
    rows = []
    for c in candidates:
        fam = charge_family(c.charge)
        e_pa = c.best_energy_hartree / c.n_atoms
        if fam == "neutral":
            e_react = neutral_formation_energy(c, mu)
            ref_txt = "Nx -> (x/2) N2"
        else:
            key = "e_n5_cation" if fam == "cation" else "e_n5_anion"
            e_ref5 = charged_refs.get(key)
            if e_ref5 is None or e_n2 is None:
                # Reference not available -- report NaN rather than a wrong number.
                e_react = float("nan")
                ref_txt = f"MISSING {key}/e_n2 (set in config)"
            else:
                e_react = charged_reaction_energy(c, e_n2, e_ref5)
                sign = "+" if fam == "cation" else "-"
                ref_txt = f"Nx{sign} -> (x-5)/2 N2 + N5{sign}"
        rows.append(dict(
            formula=c.formula, family=fam, tag=c.tag,
            charge=c.charge, n_atoms=c.n_atoms,
            e_xtb_hartree=c.e_xtb_hartree,
            e_per_atom_hartree=e_pa,
            e_reaction_hartree=e_react,
            e_reaction_ev_per_atom=(e_react / c.n_atoms) * HARTREE_TO_EV,
            e_reaction_kcalmol=e_react * HARTREE_TO_KCALMOL,
            reference_reaction=ref_txt,
            smiles=c.smiles,
            xyz=c.xyz_xtb,
            is_true_minimum=c.is_true_minimum,
            n_freq_hops=c.n_freq_hops,
            n_imaginary_freq=c.n_imaginary_freq,
        ))
    df = pd.DataFrame(rows).sort_values(["family", "n_atoms", "e_xtb_hartree"])
    # Rank within each (formula, family): 1 = most stable, ascending energy.
    df["rank"] = (df.groupby(["formula", "family"])["e_xtb_hartree"]
                    .rank(method="first").astype(int))
    return df[columns]


def resolve_charged_references(candidates: list, cfg: dict) -> dict:
    """Assemble the reference energies for the charged-family reactions.

    Priority: explicit config values (convex_hull.charged_references) win.
    Otherwise, if N5 of the relevant charge is present in the run, use its
    lowest-energy member. If neither is available for a charged family that
    IS present, raise a clear error -- the reaction energy would otherwise
    be silently wrong."""
    refs = {}
    ch = cfg.get("convex_hull", {})
    cr = ch.get("charged_references", {}) or {}

    # N2 energy: from config if given, else from the run's N2.
    refs["e_n2"] = cr.get("e_n2")
    if refs["e_n2"] is None:
        n2s = [c for c in candidates if c.n_atoms == 2 and c.charge == 0]
        if n2s:
            refs["e_n2"] = min(c.best_energy_hartree for c in n2s)

    families_present = {charge_family(c.charge) for c in candidates}
    for fam, key, q in [("cation", "e_n5_cation", 1), ("anion", "e_n5_anion", -1)]:
        if fam not in families_present:
            continue
        val = cr.get(key)
        if val is None:
            n5 = [c for c in candidates if c.n_atoms == 5 and c.charge == q]
            if n5:
                val = min(c.best_energy_hartree for c in n5)
        if val is None or refs.get("e_n2") is None:
            raise ValueError(
                f"Charged family '{fam}' is present but its reaction reference is "
                f"incomplete: need both e_n2 and {key}. Provide them under "
                f"convex_hull.charged_references in the config (mandatory pre-computed "
                f"values), or include N2 and N5{'+' if q>0 else '-'} in the run.")
        refs[key] = val
    return refs


def count_fragments(xyz_path: str, bond_threshold: float = 2.0) -> int:
    """Count connected components (fragments) in a structure, using a simple
    distance cutoff to decide which atom pairs are bonded.

    Two atoms are considered bonded if their distance is below
    bond_threshold (angstrom). A genuine N-N bond is ~1.25-1.5 A (up to ~1.6
    A when very stretched); a van der Waals N...N contact is ~3 A. A cutoff
    around 2.0 A therefore cleanly separates a real, connected molecule (one
    component) from a fragmented structure where, e.g., an N2 has detached
    (two or more components). Returns the number of components (1 = intact)."""
    import numpy as np
    try:
        lines = Path(xyz_path).read_text().splitlines()
        n = int(lines[0].split()[0])
        coords = np.array([[float(x) for x in ln.split()[1:4]]
                           for ln in lines[2:2 + n]])
    except Exception:
        return 1  # cannot read -> do not reject on this basis

    d = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if d[i, j] < bond_threshold:
                parent[find(i)] = find(j)
    return len({find(i) for i in range(n)})


def reject_fragmented(candidates: list, bond_threshold: float = 2.0) -> list:
    """Drop candidates whose optimized geometry is not a single connected
    molecule -- i.e. it fell apart during optimization into a Nx-y + Ny
    assembly (a detached N2 in van der Waals contact being the canonical
    case). Such structures are not genuine Nx allotropes and would pollute a
    training set, so they are removed after optimization/frequency steps."""
    kept, n_frag = [], 0
    for c in candidates:
        xyz = c.xyz_xtb
        if xyz and count_fragments(xyz, bond_threshold) > 1:
            n_frag += 1
            continue
        kept.append(c)
    log.info("Fragmentation filter (bond<%.1f A): removed %d fragmented "
             "structure(s), %d intact remaining.", bond_threshold, n_frag, len(kept))
    return kept


def deduplicate_candidates(candidates: list, energy_tol_kcalmol: float = 0.5) -> list:
    """Remove duplicate structures that may arise when several generators
    (isolobal / random / MAYGEN) independently produce the same topology.

    Two candidates are considered duplicates when they have the SAME
    composition and charge AND their optimized energies agree within
    energy_tol_kcalmol. Energy agreement after independent optimization is a
    robust, cheap proxy for "same minimum" here (same formula + same charge
    + same energy to <0.5 kcal/mol almost always means the same structure),
    avoiding a full 3D isomorphism/RMSD test. The lowest-energy
    representative of each duplicate group is kept.

    This runs AFTER optimization (so energies exist) but the same helper can
    be applied again after frequency verification, since following imaginary
    modes can make two initially-distinct structures collapse onto one
    minimum."""
    from collections import defaultdict
    groups = defaultdict(list)
    for c in candidates:
        if c.best_energy_hartree is None:
            continue
        groups[(c.formula, c.charge)].append(c)

    kept = []
    n_removed = 0
    for _, members in groups.items():
        members.sort(key=lambda c: c.best_energy_hartree)
        survivors = []
        for c in members:
            is_dup = any(
                abs((c.best_energy_hartree - s.best_energy_hartree) * HARTREE_TO_KCALMOL)
                < energy_tol_kcalmol
                for s in survivors)
            if is_dup:
                n_removed += 1
            else:
                survivors.append(c)
        kept.extend(survivors)
    log.info("Deduplication: removed %d duplicate structure(s), %d unique remaining.",
              n_removed, len(kept))
    return kept


def export_ranked_structures(df: pd.DataFrame, out_dir: Path) -> None:
    """Copy every retained structure into a single flat directory
    'best_structures/', renamed <formula>_<rank>.xyz with rank zero-padded
    (rank 1 = most stable of that composition). This gives a quick,
    one-folder overview of all structures ordered by stability per
    composition, without digging through the per-candidate work tree.

    Only structures that passed frequency verification as true minima are
    exported when that information is available; otherwise all are exported."""
    import shutil
    best_dir = out_dir / "best_structures"
    best_dir.mkdir(parents=True, exist_ok=True)

    exported = 0
    for _, row in df.iterrows():
        xyz = row.get("xyz")
        if not xyz or not Path(xyz).exists():
            continue
        # Prefer verified minima if the column is meaningful.
        if row.get("is_true_minimum") is False:
            continue
        formula = row["formula"]
        rank = int(row["rank"])
        dest = best_dir / f"{formula}_{rank:03d}.xyz"
        shutil.copyfile(xyz, dest)
        exported += 1
    log.info("Exported %d ranked structures to %s", exported, best_dir)


def write_summary_table(df: pd.DataFrame, out_dir: Path) -> None:
    """Write a compact human-readable summary of the species retained after
    frequency verification, one block per (family, composition), ranked by
    stability, with the reaction energy of each."""
    summary_path = out_dir / "summary_after_frequencies.csv"
    keep_cols = ["rank", "formula", "family", "charge", "n_atoms",
                 "e_reaction_kcalmol", "reference_reaction",
                 "is_true_minimum", "n_imaginary_freq", "xyz"]
    cols = [c for c in keep_cols if c in df.columns]
    verified = df
    if "is_true_minimum" in df.columns:
        verified = df[df["is_true_minimum"] != False]  # noqa: E712 (keep True and NaN)
    verified = verified.sort_values(["family", "n_atoms", "rank"])
    verified[cols].to_csv(summary_path, index=False)
    log.info("Summary of retained (verified) species written to %s", summary_path)



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
    best = df.loc[df.groupby("formula")["e_reaction_ev_per_atom"].idxmin()].copy()
    best = best.sort_values("n_atoms").reset_index(drop=True)
    pts = np.column_stack([
        best["n_atoms"].to_numpy(dtype=float),
        best["e_reaction_ev_per_atom"].to_numpy(dtype=float),
    ])
    hull_idx = set(lower_hull_1d(pts).tolist())
    best["on_hull"] = [i in hull_idx for i in range(len(best))]
    hull_pts = pts[sorted(hull_idx)]
    best["e_above_hull_ev_per_atom"] = [
        row.e_reaction_ev_per_atom - np.interp(row.n_atoms, hull_pts[:, 0], hull_pts[:, 1])
        for row in best.itertuples()
    ]
    return best


def convex_hull_binary(df: pd.DataFrame, elements: list) -> pd.DataFrame:
    """Composition-fraction hull for a two-element system, pure elements
    pinned at (0,0) and (1,0)."""
    best = df.loc[df.groupby("formula")["e_reaction_hartree"].idxmin()].copy()

    def frac_b(row):
        comp = parse_formula(row["formula"])
        na, nb = comp.get(elements[0], 0), comp.get(elements[1], 0)
        return nb / (na + nb) if (na + nb) else 0.0

    best["x_frac"] = best.apply(frac_b, axis=1)
    pts = np.column_stack([
        np.concatenate([[0.0, 1.0], best["x_frac"].to_numpy()]),
        np.concatenate([[0.0, 0.0], best["e_reaction_ev_per_atom"].to_numpy()]),
    ])
    hull_idx = set(lower_hull_1d(pts).tolist()) - {0, 1}
    best["on_hull"] = [(i + 2) in hull_idx for i in range(len(best))]
    hull_pts = pts[sorted(hull_idx | {0, 1})]
    best["e_above_hull_ev_per_atom"] = [
        row.e_reaction_ev_per_atom - np.interp(row.x_frac, hull_pts[:, 0], hull_pts[:, 1])
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
        x, y = hull_df["n_atoms"], hull_df["e_reaction_ev_per_atom"]
        xlabel, ylabel = "Number of atoms", f"Formation energy per atom vs {ref_formula} (eV)"
    else:
        x, y = hull_df["x_frac"], hull_df["e_reaction_ev_per_atom"]
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
    parser = argparse.ArgumentParser(
        description="polyN pipeline: generate/optimize/verify/rank polynitrogen "
                    "allotropes (neutral, cation, anion) at GFN2-xTB with frequency "
                    "verification, and build per-charge-family convex hulls.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Topology generators (choose one or more with --generators; combining them
gives a richer, deduplicated candidate pool):

  isolobal   (DEFAULT)  PubChem hydrocarbon harvest + isolobal C->N
                        substitution. Most chemically-motivated source.
  random                Random valence-correct molecular-graph sampling
                        (any size/charge, incl. localized charge motifs).
  maygen                Exhaustive constitutional-isomer enumeration via an
                        external MAYGEN jar (neutral families only).
  geng                  Exhaustive connected-graph enumeration via geng
                        (nauty) + bond-order assignment. Formal charges are
                        assigned at the graph stage, so neutral/cation/anion
                        families all emerge from the same graph set with no
                        phantom-hydrogen saturation. Requires nauty's geng.

Examples:
  # default: isolobal only
  python3 polyN_pipeline.py --config config_seeds.yaml
  # combine isolobal + exhaustive geng enumeration
  python3 polyN_pipeline.py --config config.yaml --generators isolobal geng
  # use pre-existing input files, run no generator
  python3 polyN_pipeline.py --config config.yaml --generators none
""")
    parser.add_argument("--config", required=True, help="YAML config file")
    parser.add_argument("--generators", nargs="+", default=["isolobal"],
                         choices=["isolobal", "random", "maygen", "geng", "none"],
                         help="Topology generator(s) to run before optimization "
                              "(default: isolobal). Use 'none' to rely solely on "
                              "pre-existing input files in input_dir.")
    parser.add_argument("--skip-freq", action="store_true",
                         help="Force-skip the frequency verification step even if enabled")
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    out_dir = Path(cfg.get("output_dir", "./results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = out_dir / "work"
    work_dir.mkdir(exist_ok=True)

    generators = [g for g in args.generators if g != "none"]

    log.info("== Step 1/5: topology generation & reading ==")
    run_generators(cfg, generators, work_dir)
    candidates = gather_candidates(cfg)
    log.info("Total candidates: %d", len(candidates))

    log.info("== Step 2/5: GFN2-xTB optimization (checkpoint + refinement) ==")
    candidates = optimize_all_xtb(candidates, work_dir, cfg["xtb"])

    if cfg.get("deduplicate", True):
        log.info("Deduplicating candidates (post-optimization)...")
        candidates = deduplicate_candidates(
            candidates, cfg.get("dedup_tol_kcalmol", 0.5))

    log.info("== Step 3/5: selecting lowest-energy structures per stoichiometry ==")
    selected = select_lowest(candidates, cfg["selection"])

    do_freq = cfg.get("frequency", {}).get("enabled", True) and not args.skip_freq
    if do_freq:
        log.info("== Step 4/5: frequency verification (true minima only) ==")
        selected = verify_all_minima(selected, work_dir, cfg)
        if cfg.get("deduplicate", True):
            # Following imaginary modes can collapse initially-distinct
            # structures onto the same minimum -- deduplicate again.
            log.info("Deduplicating verified minima (post-frequency)...")
            selected = deduplicate_candidates(
                selected, cfg.get("dedup_tol_kcalmol", 0.5))
    else:
        log.info("== Step 4/5: skipped (frequency verification disabled) ==")

    # Reject fragmented structures (e.g. a detached N2 in van der Waals
    # contact) -- these are not genuine single Nx molecules. Done last, since
    # optimization AND imaginary-mode following can both cause fragmentation.
    if cfg.get("reject_fragmented", True):
        selected = reject_fragmented(
            selected, cfg.get("fragment_bond_threshold", 2.0))

    log.info("== Step 5/5: references & convex hulls (per charge family) ==")
    mu = compute_neutral_reference(cfg, work_dir)
    charged_refs = resolve_charged_references(selected, cfg)
    df = build_results_table(selected, mu, charged_refs)
    df.to_csv(out_dir / "results.csv", index=False)
    log.info("Full results table written to %s", out_dir / "results.csv")

    # Compact summary of retained (verified) species + ranked flat directory.
    write_summary_table(df, out_dir)
    export_ranked_structures(df, out_dir)

    hulls, mode = build_convex_hulls_by_family(df, cfg)
    ref_formula = cfg["convex_hull"].get("reference", {}).get("formula", "N2")

    log.info("Done. Stable stoichiometries (on hull), by charge family:")
    for family, hull_df in hulls.items():
        hull_df.to_csv(out_dir / f"convex_hull_{family}.csv", index=False)
        plot_hull(hull_df, mode, out_dir / f"convex_hull_{family}.png", ref_formula, family=family)
        if family == "neutral":
            ref_note = "Nx -> (x/2) N2"
        else:
            sign = "+" if family == "cation" else "-"
            ref_note = f"Nx{sign} -> (x-5)/2 N2 + N5{sign}"
        log.info("  [%s] (reference reaction: %s)", family, ref_note)
        for _, row in hull_df[hull_df["on_hull"]].iterrows():
            log.info("    %s", row["formula"])


if __name__ == "__main__":
    main()
