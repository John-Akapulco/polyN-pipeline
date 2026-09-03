#!/usr/bin/env python3
"""
harvest_hydrocarbons.py
========================

Mine PubChem for EXPERIMENTAL hydrocarbon structures (CxHy, any degree of
saturation/unsaturation) across a range of carbon counts, to feed into
cxhx_to_nx.py's isolobal C->N substitution (--smiles-file mode).

IMPORTANT: this script could not be executed or tested in the environment
that wrote it (no network access to pubchem.ncbi.nlm.nih.gov from that
sandbox). The PubChemPy API calls below are written according to the
documented interface, but you should sanity-check the first few results
before committing to a large run -- see the CHECKS TO DO section below.

CHECKS TO DO on your first run
-------------------------------
1. Confirm formula search returns EXACT formula matches, not compounds
   that merely CONTAIN CxHy as a substructure with extra atoms. PubChemPy's
   formula search may need an explicit exact-match option depending on the
   PUG REST version -- if you see non-hydrocarbon elements in the output,
   add server-side filtering or rely on cxhx_to_nx.py's own rejection of
   non-C/H atoms (it already skips anything that isn't a pure hydrocarbon,
   so this is a safety net either way, just less efficient).
2. Confirm the request rate stays comfortably under PubChem's usage policy
   (they ask for no more than ~5 requests/second and recommend spacing out
   large jobs). The REQUEST_DELAY_S below is a conservative default --
   raise it if you see HTTP 503 / throttling errors in practice.
3. Some formulas (e.g. C10H16, common in terpenes) can return thousands of
   hits -- MAX_PER_FORMULA below caps this; raise cautiously.

Installation
------------
    pip install pubchempy

Usage
-----
    python harvest_hydrocarbons.py --min-c 4 --max-c 16 -o hydrocarbons_raw.smi
    python cxhx_to_nx.py --smiles-file hydrocarbons_raw.smi --max-abs-charge 1 \\
        -o maygen_output/seeds/
"""

from __future__ import annotations

import argparse
import socket
import time
from pathlib import Path

try:
    import pubchempy as pcp
except ImportError:
    raise SystemExit("pubchempy is required: pip install pubchempy")

# HARD network timeout: without this, a hung/dropped connection to PubChem
# can block get_compounds() INDEFINITELY (observed in practice: a process
# left running for 13 hours accumulated only ~7 seconds of CPU time --
# stuck waiting on a single request that never returned or errored out).
# This affects all sockets process-wide, which is blunt but effective.
SOCKET_TIMEOUT_S = 30
socket.setdefaulttimeout(SOCKET_TIMEOUT_S)

REQUEST_DELAY_S = 0.3     # conservative spacing between PubChem requests
MAX_PER_FORMULA = 300     # cap per (C count, H count) formula to bound run time
MAX_RETRIES = 3


def formulas_to_try(n_c: int):
    """H counts to scan for a given carbon count, covering the full range
    from a fully unsaturated cage/ring (few H) to a fully saturated acyclic
    or cyclic alkane (many H). Only even H counts are chemically possible
    for a plain CxHy hydrocarbon (closed-shell, no radicals)."""
    h_min = 2                 # very unsaturated / highly caged (e.g. tetrahedrane-like, CxH~x or less)
    h_max = 2 * n_c + 2        # acyclic saturated alkane CnH2n+2 (upper bound)
    return [h for h in range(h_min, h_max + 1, 2)]


def fetch_formula(formula: str, max_per_formula: int) -> list:
    """Fetch up to max_per_formula compounds matching an exact molecular
    formula, with basic retry on transient errors."""
    for attempt in range(MAX_RETRIES):
        try:
            compounds = pcp.get_compounds(formula, "formula", listkey_count=max_per_formula)
            return compounds
        except Exception as exc:
            wait = REQUEST_DELAY_S * (2 ** attempt)
            print(f"  [retry {attempt+1}/{MAX_RETRIES}] {formula}: {exc} -- waiting {wait:.1f}s")
            time.sleep(wait)
    print(f"  [FAILED] {formula}: giving up after {MAX_RETRIES} attempts")
    return []


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-c", type=int, default=4, help="Minimum carbon count (default: 4)")
    parser.add_argument("--max-c", type=int, default=16, help="Maximum carbon count (default: 16)")
    parser.add_argument("--max-per-formula", type=int, default=MAX_PER_FORMULA,
                         help=f"Cap on compounds fetched per exact formula (default: {MAX_PER_FORMULA})")
    parser.add_argument("-o", "--output", required=True, help="Output .smi file (one SMILES per line)")
    parser.add_argument("--resume", action="store_true",
                         help="Resume from the last completed formula recorded in <output>.progress "
                              "(use this after killing a hung/interrupted run instead of starting over)")
    args = parser.parse_args()

    out_path = Path(args.output)
    progress_path = out_path.with_suffix(out_path.suffix + ".progress")

    seen_smiles = set()
    done_formulas = set()
    if args.resume:
        if out_path.exists():
            with open(out_path) as fh:
                seen_smiles = {line.strip() for line in fh if line.strip()}
            print(f"Resuming: {len(seen_smiles)} SMILES already collected.")
        if progress_path.exists():
            with open(progress_path) as fh:
                done_formulas = {line.strip() for line in fh if line.strip()}
            print(f"Resuming: {len(done_formulas)} formula(s) already completed, will be skipped.")

    mode = "a" if args.resume and out_path.exists() else "w"
    with open(out_path, mode) as fh, open(progress_path, "a" if args.resume else "w") as progress_fh:
        for n_c in range(args.min_c, args.max_c + 1):
            for n_h in formulas_to_try(n_c):
                formula = f"C{n_c}H{n_h}"
                if formula in done_formulas:
                    continue
                compounds = fetch_formula(formula, args.max_per_formula)
                n_new = 0
                for c in compounds:
                    smi = getattr(c, "canonical_smiles", None) or getattr(c, "connectivity_smiles", None)
                    if not smi or smi in seen_smiles:
                        continue
                    seen_smiles.add(smi)
                    fh.write(smi + "\n")
                    n_new += 1
                fh.flush()
                if compounds:
                    print(f"{formula}: {len(compounds)} found, {n_new} new -> total {len(seen_smiles)}")
                # Record this formula as done ONLY after it fully completed,
                # so a kill/hang mid-formula correctly retries it on resume.
                progress_fh.write(formula + "\n")
                progress_fh.flush()
                time.sleep(REQUEST_DELAY_S)

    print(f"\nDone. {len(seen_smiles)} unique hydrocarbon SMILES written to {out_path}")
    print(f"Next step:\n"
          f"  python cxhx_to_nx.py --smiles-file {out_path} --max-abs-charge 1 -o maygen_output/seeds/")


if __name__ == "__main__":
    main()
