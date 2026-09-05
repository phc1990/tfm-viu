#!/usr/bin/env bash

# Sequential DS9 review of frame-specific Horizons overlays.
# Intended location:
#   tfm-viu/python/src/plots/open_ds9_multiple.sh
#
# Run from repository root:
#   python/src/plots/open_ds9_multiple.sh
#
# Close each DS9 window to advance to the next frame.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Resolve repository root robustly.
ROOT_DIR="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
if [ -z "$ROOT_DIR" ]; then
    ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
fi

cd "$ROOT_DIR" || exit 1

HORIZONS_DIR="$ROOT_DIR/results/horizons_validation"
TEMP_DIR="$ROOT_DIR/temp"

# IMPORTANT:
# In this machine, "ds9" is a shell alias:
#   /Users/eracero/mysoftware/SAOImageDS9.app/Contents/MacOS/ds9
#
# Aliases are not expanded inside this non-interactive script, so we
# invoke the real executable explicitly.
DS9_BIN="/Users/eracero/mysoftware/SAOImageDS9.app/Contents/MacOS/ds9"

if [ ! -x "$DS9_BIN" ]; then
    echo "[ERROR] DS9 executable not found or not executable:"
    echo "        $DS9_BIN"
    exit 127
fi

open_ds9() {
    local obj="$1"
    local obsid="$2"
    local frame="$3"

    local reg=""
    local image_base=""
    local fits=""

    reg="$(find "$HORIZONS_DIR/$obj/$obsid" \
        -type f \
        -name "*${frame}*_horizons.reg" \
        -print -quit 2>/dev/null)"

    echo
    echo "============================================================"
    echo "$obj  $obsid  $frame"

    if [ -z "$reg" ]; then
        echo "REG : NOT FOUND"
        echo "============================================================"
        echo "[SKIP] No existe region de Horizons para este frame."
        return 0
    fi

    image_base="$(basename "$reg" _horizons.reg)"

    fits="$(find "$TEMP_DIR/$obsid" \
        -type f \
        -name "${image_base}.FTZ" \
        -print -quit 2>/dev/null)"

    echo "FITS: ${fits:-NOT FOUND}"
    echo "REG : $reg"
    echo "============================================================"

    if [ -z "$fits" ]; then
        echo "[SKIP] No encuentro la imagen exacta:"
        echo "       ${image_base}.FTZ"
        return 0
    fi

    # Deliberately no '&': close DS9 to continue to the next frame.
    "$DS9_BIN" "$fits" \
        -scale mode zscale \
        -regions load "$reg" \
        -zoom to fit \
        -title "$obj $obsid $frame"
}

echo
echo "Sequential Horizons/DS9 review"
echo "Script dir : $SCRIPT_DIR"
echo "Repository : $ROOT_DIR"
echo "Horizons   : $HORIZONS_DIR"
echo "Temp       : $TEMP_DIR"
echo "DS9        : $DS9_BIN"
echo "Close each DS9 window to advance."
echo

# open_ds9 Landi 0700182001 OMS008

# for frame in \
#     OMS006 OMS007 OMS008 OMS009 OMS010 OMS011 \
#     OMS012 OMS013 OMS014 OMS015 OMS016 OMS017 \
#     OMS018 OMS019 OMS020 OMS021 OMS022 OMS023
# do
#     open_ds9 Pobeda 0305540501 "$frame"
# done

# open_ds9 Weisell 0691070301 OMS401
# open_ds9 Kaneko 0160961001 OMS004
# open_ds9 Asteropaios 0803161101 OMS411
# open_ds9 Mucha 0300240101 OMS008
# open_ds9 The 0501270301 OMS007
# open_ds9 Gurzhij 0744490401 OMS434
# open_ds9 Richardbaum 0201902401 OMS404
# open_ds9 Ivanka 0692330401 OMS007
# open_ds9 Pesonen 0148880101 OMS408
# open_ds9 Jepejacobsen 0784401201 OMS006
# open_ds9 Astrometria 0793581001 OMS006

# open_ds9 Potomac 0674480401 OMS407
# open_ds9 Sarpedon 0800400501 OMS408

# open_ds9 Tinette 0400890201 OMS409
# open_ds9 Tinette 0400890201 OMS412

# open_ds9 Seeligeria 0303670101 OMS413
# open_ds9 Seeligeria 0303670101 OMS009

# open_ds9 Vivian 0693990301 OMS011
# open_ds9 Vivian 0693990301 OMS015

# open_ds9 Flagstaff 0804250301 OMS008
# open_ds9 Epstein 0205330501 OMS401
# open_ds9 Kalman 0725290145 OMS416
# open_ds9 Parler 0747400143 OMS412

# open_ds9 "1996_UT" 0412592401 OMS010
# open_ds9 "1996_BB2" 0694640901 OMS404


# open_ds9 Richardbaum 0201902401 OMS404

open_ds9 Dowling 0673002335 OMS409

open_ds9 Brouwer 0801681301 OMS006
open_ds9 Brouwer 0801681301 OMS007

open_ds9 Ada 0803030301 OMS006
open_ds9 Ada 0803030301 OMS007

open_ds9 Tampere 0740920301 OMS406

open_ds9 "1996_UT" 0412592501 OMS006

open_ds9 Chrisclark 0670120401 OMS413

echo
echo "============================================================"
echo "Review finished."
echo "============================================================"
