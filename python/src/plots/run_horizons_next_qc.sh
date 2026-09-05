#!/usr/bin/env bash

# Horizons QC additions + rows present in colleague screening (Y/D)
# but absent from photometry_output.
#
# Intended location:
#   tfm-viu/python/src/plots/run_horizons_next.sh
#
# Usage:
#   python/src/plots/run_horizons_next.sh
#
# If the colleague screening is a separate CSV, pass it as the first arg:
#   python/src/plots/run_horizons_next.sh path/to/screening_companero.csv

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
if [ -z "$ROOT_DIR" ]; then
    ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
fi
cd "$ROOT_DIR" || exit 1

SCRIPT="python/src/plots/recompute_horizons_frame_for_object.py"
MAIN_SCREENING="results/batch_L/elena/screening.csv"
OUTDIR="results/horizons_validation"
COLLEAGUE_SCREENING="${1:-}"

run_horizons() {
    local screening="$1"
    local target="$2"
    local obsid="$3"
    local fits_name="$4"

    echo
    echo "============================================================"
    echo "HORIZONS: $target  $obsid"
    echo "FITS    : $fits_name"
    echo "SCREEN  : $screening"
    echo "============================================================"

    python3 "$SCRIPT" "$target" \
        --obsid "$obsid" \
        --fits-name "$fits_name" \
        --screening "$screening" \
        --output-dir "$OUTDIR" \
        --horizons-location @XMM

    rc=$?
    if [ $rc -ne 0 ]; then
        echo "[WARN] Horizons failed for '$target' / '$fits_name' (exit $rc); continuing."
    fi
}

echo
echo "Repository : $ROOT_DIR"
echo "Main screen: $MAIN_SCREENING"
echo "Output     : $OUTDIR"

# ------------------------------------------------------------------
# QC additions from photometry_output:
# explicit frames only, to avoid rerunning unrelated rows.
# ------------------------------------------------------------------

# Reconfirm end position / marginal source.
run_horizons "$MAIN_SCREENING" Richardbaum 0201902401 \
    P0201902401OMS404SIMAGE1000.FTZ

# Extreme UVW1-Vpred outlier.
run_horizons "$MAIN_SCREENING" Dowling 0673002335 \
    P0673002335OMS409SIMAGE0000.FTZ

# Strong internal disagreement between the two frames.
run_horizons "$MAIN_SCREENING" Brouwer 0801681301 \
    P0801681301OMS006FSIMAGL000.FTZ
run_horizons "$MAIN_SCREENING" Brouwer 0801681301 \
    P0801681301OMS007FSIMAGL000.FTZ

# ~3-sigma internal frame discrepancy.
run_horizons "$MAIN_SCREENING" Ada 0803030301 \
    P0803030301OMS006FSIMAGL000.FTZ
run_horizons "$MAIN_SCREENING" Ada 0803030301 \
    P0803030301OMS007FSIMAGL000.FTZ

# Very close to the nominal limiting magnitude.
run_horizons "$MAIN_SCREENING" Tampere 0740920301 \
    P0740920301OMS406SIMAGE1000.FTZ

# Second 1996 UT observation: marginal and not in the previous OMS010 check.
run_horizons "$MAIN_SCREENING" "1996 UT" 0412592501 \
    P0412592501OMS006FSIMAGL000.FTZ

# Blue-side colour outlier, lower priority but worth a positional check.
run_horizons "$MAIN_SCREENING" Chrisclark 0670120401 \
    P0670120401OMS413SIMAGE1000.FTZ


echo
echo "============================================================"
echo "Finished."
echo "============================================================"
