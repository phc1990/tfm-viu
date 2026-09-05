#!/usr/bin/env bash

# Run Horizons recomputation for the next UVW1 QC targets.
# Intended location:
#   tfm-viu/python/src/plots/run_horizons_next.sh
#
# Run from repository root:
#   python/src/plots/run_horizons_next.sh

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
if [ -z "$ROOT_DIR" ]; then
    ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
fi
cd "$ROOT_DIR" || exit 1

SCRIPT="python/src/plots/recompute_horizons_frame_for_object.py"
SCREENING="results/batch_L/elena/screening.csv"
OUTDIR="results/horizons_validation"

targets=(
  "Ivanka"
  "Pesonen"
  "Jepejacobsen"
  "Astrometria"
  "Potomac"
  "Sarpedon"
  "Tinette"
  "Seeligeria"
  "Vivian"
  "Flagstaff"
  "Epstein"
  "Kalman"
  "Parler"
  "1996 BB2"
)

echo "Repository: $ROOT_DIR"
echo "Screening : $SCREENING"
echo "Output    : $OUTDIR"
echo

for target in "${targets[@]}"; do
    echo
    echo "============================================================"
    echo "HORIZONS: $target"
    echo "============================================================"

    python3 "$SCRIPT" "$target" \
      --screening "$SCREENING" \
      --output-dir "$OUTDIR" \
      --horizons-location @XMM

    rc=$?
    if [ $rc -ne 0 ]; then
        echo "[WARN] Horizons failed for '$target' (exit $rc); continuing."
    fi
done

# User specifically requested the OMS010 case in OBSID 0412592401.
echo
echo "============================================================"
echo "HORIZONS: 1996 UT / 0412592401"
echo "============================================================"

python3 "$SCRIPT" "1996 UT" \
  --obsid 0412592401 \
  --screening "$SCREENING" \
  --output-dir "$OUTDIR" \
  --horizons-location @XMM

rc=$?
if [ $rc -ne 0 ]; then
    echo "[WARN] Horizons failed for '1996 UT' / 0412592401 (exit $rc)."
fi

echo
echo "============================================================"
echo "Finished."
echo "============================================================"
