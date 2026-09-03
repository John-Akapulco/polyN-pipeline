#!/usr/bin/env bash
# =============================================================================
# prepare_and_run_polyN.sh
#
# 1) Generates MAYGEN isomers for the neutral (even-N) family.
# 2) Generates manual starting topologies (ring + open chain) for the
#    odd-N family, later reused for both the cation and anion entries in
#    the pipeline config (MAYGEN's own enumeration / -setElements crashes
#    on single-element odd-degree-2 systems -- see the manual, Sec 9.1).
# 3) Validates every generated SMILES with RDKit before any heavy calculation.
# 4) Optionally launches polyN_pipeline.py.
#
# Usage:
#   ./prepare_and_run_polyN.sh [options]
#
# Options:
#   -j PATH   Path to MAYGEN-1.8.jar (default: ~/MAYGEN/target/MAYGEN-1.8.jar)
#   -o DIR    MAYGEN/topology output directory (default: ./maygen_output)
#   -p PATH   Python interpreter to use (default: python3)
#   -c FILE   Pipeline config file (default: config_example.yaml)
#   -r        Also run the pipeline at the end (python3 polyN_pipeline.py --config ... --skip-dftb)
#   -h        Show this help and exit
#
# Example:
#   ./prepare_and_run_polyN.sh -p ~/miniconda3/bin/python3 -r
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults / argument parsing
# ---------------------------------------------------------------------------
MAYGEN_JAR="$HOME/MAYGEN/target/MAYGEN-1.8.jar"
OUT_DIR="./maygen_output"
PYTHON_BIN="python3"
CONFIG_FILE="config_example.yaml"
RUN_PIPELINE=0

usage() { sed -n '2,25p' "$0"; }

while getopts "j:o:p:c:rh" opt; do
  case "$opt" in
    j) MAYGEN_JAR="$OPTARG" ;;
    o) OUT_DIR="$OPTARG" ;;
    p) PYTHON_BIN="$OPTARG" ;;
    c) CONFIG_FILE="$OPTARG" ;;
    r) RUN_PIPELINE=1 ;;
    h) usage; exit 0 ;;
    *) usage; exit 1 ;;
  esac
done

EVEN_N=(2 4 6 8 10 12 14 16)   # neutral family
ODD_N=(3 5 7 9 11 13 15)       # cation / anion families (shared topology files)

echo "== MAYGEN jar   : $MAYGEN_JAR"
echo "== Output dir   : $OUT_DIR"
echo "== Python       : $PYTHON_BIN"
echo "== Config file  : $CONFIG_FILE"
echo

mkdir -p "$OUT_DIR"

# ---------------------------------------------------------------------------
# 1) Neutral family (even N) -- via MAYGEN
# ---------------------------------------------------------------------------
if [[ ! -f "$MAYGEN_JAR" ]]; then
  echo "ERROR: MAYGEN jar not found at $MAYGEN_JAR (use -j to point to it)." >&2
  exit 1
fi

echo "-- Generating neutral (even-N) isomers with MAYGEN --"
for n in "${EVEN_N[@]}"; do
  out_file="$OUT_DIR/N${n}.smi"
  if [[ -s "$out_file" ]]; then
    echo "   N${n}.smi already exists, skipping."
    continue
  fi
  java -jar "$MAYGEN_JAR" -f "N${n}" -smi -o "$OUT_DIR/"
  echo "   N${n}: $(wc -l < "$out_file" | tr -d ' ') isomer(s)"
done
echo

# ---------------------------------------------------------------------------
# 2) Odd-N family (cation/anion precursor topologies) -- manual SMILES
#    MAYGEN's default valence-3 enumeration gives zero isomers for odd N
#    (handshake-lemma parity), and its -setElements degree-2 path crashes
#    (ArrayIndexOutOfBoundsException in degree2graph). We write two
#    complementary starting topologies by hand instead:
#      - a simple ring (all single bonds)
#      - an open chain (all single bonds)
#    Both are just starting guesses for xtb; the correct charge/spin is
#    applied later at the tblite/DFTB+ level, not here.
# ---------------------------------------------------------------------------
echo "-- Writing manual ring + chain topologies for odd-N systems --"
for n in "${ODD_N[@]}"; do
  out_file="$OUT_DIR/N${n}.smi"
  if [[ -s "$out_file" ]]; then
    echo "   N${n}.smi already exists, skipping."
    continue
  fi

  # Ring: [N]1[N][N]...[N]1  (n atoms)
  ring="[N]1"
  for ((i = 1; i < n; i++)); do ring+="[N]"; done
  ring+="1"

  # Open chain: [N][N][N]...[N]  (n atoms)
  chain="[N]"
  for ((i = 1; i < n; i++)); do chain+="[N]"; done

  { echo "$ring"; echo "$chain"; } > "$out_file"
  echo "   N${n}: wrote ring + chain topologies"
done
echo

# ---------------------------------------------------------------------------
# 3) Validate every .smi file with RDKit before any heavy calculation
# ---------------------------------------------------------------------------
echo "-- Validating all SMILES with RDKit --"
"$PYTHON_BIN" - "$OUT_DIR" <<'PYEOF'
import sys
from pathlib import Path
from rdkit import Chem

out_dir = Path(sys.argv[1])
total_bad = 0
for f in sorted(out_dir.glob("*.smi")):
    lines = [l.strip() for l in open(f) if l.strip()]
    bad = [s for s in lines if Chem.MolFromSmiles(s) is None]
    total_bad += len(bad)
    print(f"  {f.name:10s} -> {len(lines):3d} isomer(s), {len(bad)} invalid")
    for s in bad[:5]:
        print(f"       INVALID: {s}")

if total_bad:
    print(f"\n{total_bad} invalid SMILES found -- fix before running the pipeline.")
    sys.exit(1)
print("\nAll SMILES are valid.")
PYEOF
echo

# ---------------------------------------------------------------------------
# 4) Optionally launch the pipeline
# ---------------------------------------------------------------------------
if [[ "$RUN_PIPELINE" -eq 1 ]]; then
  if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: config file '$CONFIG_FILE' not found (use -c to point to it)." >&2
    exit 1
  fi
  echo "-- Launching polyN_pipeline.py --"
  "$PYTHON_BIN" polyN_pipeline.py --config "$CONFIG_FILE" --skip-dftb
else
  echo "Preparation complete. To run the pipeline:"
  echo "  $PYTHON_BIN polyN_pipeline.py --config $CONFIG_FILE --skip-dftb"
  echo "(or re-run this script with -r to do it automatically)"
fi
