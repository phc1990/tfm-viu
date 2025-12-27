#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
expo_photometry.py

Trail photometry using a *single exposure* image (no mosaic sky image),
but still calibrated against the OM source-list / OBSMLI to obtain
the AB zero point (and zero-point uncertainty if available).

CLI interface is intentionally the same as photometry.py so that
screening.py can call this script with the same arguments.

Main differences vs photometry.py:
- Does NOT download or use the mosaic sky image.
- Always uses the input --fits exposure file for the UI / trail selection.
- Uses the units-aware PhotTable.perform_trail_photometry from phot.py.
- CSV row includes net_counts / net_counts_err (from the COUNTS branch).
"""

from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
from configparser import ConfigParser

from src.photometry.hdu import HDUW
from src.photometry.ui import UI, TrailSelector
from src.photometry.phot import PhotTable
from src.photometry.phot import PhotometryResult

from common import OBS_ID_COLS, TARGET_COLS, FILTER_COLS, FITS_FILE_COLS, POS1_DEC_COLS, POS1_RA_COLS, POS2_DEC_COLS, POS2_RA_COLS
from common import extract_row_value, append_row

# Try to reuse helpers from photometry.py (download root, WCS, PNG, CSV, etc.)
try:
    from photometry import (
        _normalize_selection_to_ap_ann,
    )
    HAVE_PHOTOMETRY_HELPERS = True

except Exception:
    HAVE_PHOTOMETRY_HELPERS = False
    # --- Fallback local implementations (used only if import above fails) ---



def resolve_photometry_csv_from_ini(ini_path):
    """Return (csv_path, include_headers) from [PHOTOMETRY_RESULTS] in screening.ini."""
    if not ini_path:
        return Path("expo_photometry.csv"), True

    ini = Path(ini_path).expanduser()
    if not ini.exists():
        return Path("expo_photometry.csv"), True

    cfg = ConfigParser()
    cfg.read(ini)

    if cfg.has_section("PHOTOMETRY_RESULTS"):
        csv_path = Path(
            cfg.get("PHOTOMETRY_RESULTS", "CSV_FILEPATH",
                    fallback="expo_photometry.csv")
        ).expanduser()
        include_headers = cfg.getboolean(
            "PHOTOMETRY_RESULTS",
            "INCLUDE_HEADERS",
            fallback=True
        )
        return csv_path, include_headers

    return Path("expo_photometry.csv"), True


# -----------------------------------------------------------------------------
def zp_from_ini_for_filter(config: ConfigParser, filter: str) -> float:
    """
    If screening.ini defines [PHOTOMETRY_RESULTS] ABM0<BAND> for this filter,
    return (zp, keyword). Otherwise (None, None).

    Example:
      filter L  → band UVW1 → key ABM0UVW1
    """
    # Use the same mapping as PhotTable: L→UVW1, M→UVM2, S→UVW2, etc.
    band = PhotTable._filter_to_band(filter)
    if not band:
        return None, None

    key = f"ABM0{band}"
    raw = config["PHOTOMETRY"][key]
    return float(raw)
        

def _build_csv_row(
    target_name: str,
    obs_id: str,
    filt: str,
    fits_name: str,
    result,
    selector: TrailSelector,
) -> dict[str, Any]:
    """
    Build a CSV row with both photometric and geometric information.

    If `result` is the string "null", None, or otherwise invalid,
    fill all photometry-related fields with None so that the CSV
    row is still valid and complete.
    """

    # -------------------------------------------
    # Handle "null" result (no photometry available)
    # -------------------------------------------
    if result is None or result == "null":
        return {
            "target_name": target_name,
            "observation_id": obs_id,
            "filter": filt,
            "fits_name": fits_name,

            # Photometry fields → None
            "mag_ab": None,
            "mag_err": None,
            "count_rate": None,
            "count_rate_err": None,
            "net_counts": None,
            "net_counts_err": None,
            "aper_counts": None,
            "bkg_per_pix": None,
            "bkg_rms_per_pix": None,
            "A_ap_eff": None,
            "A_bg_eff": None,
            "zp_ab": None,

            # Geometry from selector
            "trail_height_pix": getattr(selector, "height", None),
            "trail_semi_out_pix": getattr(selector, "semi_out", None),
            "trail_semi_in_pix": getattr(selector, "semi_in", None),
        }

    # -------------------------------------------
    # Normal case: result contains valid photometry
    # -------------------------------------------
    row: dict[str, Any] = {
        "target_name": target_name,
        "observation_id": obs_id,
        "filter": filt,
        "fits_name": fits_name,

        # Magnitudes
        "mag_ab": result.mag_ab,
        "mag_err": result.mag_err,

        # Rates (optional depending on image type)
        "count_rate": result.count_rate,
        "count_rate_err": result.count_rate_err,

        # Counts
        "net_counts": result.net_counts,
        "net_counts_err": result.net_counts_err,
        "aper_counts": result.aper_counts,

        # Background and areas
        "bkg_per_pix": result.bkg_per_pix,
        "bkg_rms_per_pix": result.bkg_rms_per_pix,
        "A_ap_eff": result.A_ap_eff,
        "A_bg_eff": result.A_bg_eff,

        # Zero point provenance
        "zp_ab": result.zp_ab,

        # Geometry of trail aperture
        "trail_height_pix": getattr(selector, "height", None),
        "trail_semi_out_pix": getattr(selector, "semi_out", None),
        "trail_semi_in_pix": getattr(selector, "semi_in", None),
    }

    return row


def _build_screenshot_path(
    config: ConfigParser,
    fits_path: Path,
    target: str,
) -> Path:
    """
    Build PNG output path for a given FITS + target.

    - If screening.ini provides [PHOTOMETRY_RESULTS] PNG_FILEPATH,
      use that directory.
    - Otherwise fall back to fits_path.parent (previous behaviour).
    - If the target PNG already exists, append __1, __2, ... before
      the .png extension until a free filename is found.
    """
    # 1) Decide base directory
    base_dir = config['SCREENING']['SCREENSHOOTS_DIRECTORY']

    # 2) Base name without numeric suffix
    target_safe_name: str = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in (target or ""))
    base_name = f"{fits_path.stem}__{target_safe_name}"

    # 3) First candidate: no numeric suffix
    candidate = base_dir / f"{base_name}.png"

    # 4) If file exists, append __x where x = 1, 2, ...
    if candidate.exists():
        i = 1
        while True:
            candidate = base_dir / f"{base_name}__{i}.png"
            if not candidate.exists():
                break
            i += 1

    return candidate


def action_photometry(
    config: ConfigParser,
    screening_row: dict[str, Any]
) -> None:
    
    fits_path = Path(extract_row_value(screening_row, FITS_FILE_COLS))
    if not fits_path.exists():
        raise FileNotFoundError(f"Exposure FITS not found: {fits_path}")

    observation_id: str = extract_row_value(screening_row, OBS_ID_COLS)
    target: str = extract_row_value(screening_row, TARGET_COLS)
    filter = extract_row_value(screening_row, FILTER_COLS).upper()
    ra1: float = float(extract_row_value(screening_row, POS1_RA_COLS))
    dec1: float = float(extract_row_value(screening_row, POS1_DEC_COLS))
    ra2: float = float(extract_row_value(screening_row, POS2_RA_COLS))
    dec2: float = float(extract_row_value(screening_row, POS2_DEC_COLS))

    # (rest of your ZP / OBSMLI / HDUW / UI / trail selection logic goes here)
    # ZP from INI if available (previous step you added)
    zp_ini: float = zp_from_ini_for_filter(
        config=config,
        filter=filter,
    )

    # Where to store/download ancillary products (OBSMLI, etc.)
    # dest_root = _resolve_download_root_from_ini(ini_arg)
    # dest_dir = _dest_dir_for(obs_id, filt, dest_root)
    # srclist_path = None
    # srckind = None
    # if zp_ini is None:
    #     srclist_path, srckind = ensure_om_srclist_or_obsmlist(
    #         str(observation_id),
    #         force=False,
    #         dest_root=dest_dir,
    #     )

    # No mosaic sky image here: always operate on the exposure
    fits_name = fits_path.name
    
    # --- Build HDUW / PhotTable, set zero point ---
    hduw = HDUW(fits_path)
    hduw = HDUW(file=str(fits_path))
    arr = np.asarray(
        getattr(hduw, "data", getattr(hduw.hdu, "data", None)),
        dtype=float
    )
    if arr is None or arr.ndim != 2:
        raise ValueError("UI: expected a 2-D image in the HDU.")

    arr = np.ma.masked_invalid(arr)  # what you’re displaying
    pt = PhotTable(hduw)
    pt.zero_point = zp_ini
    # --- Force COUNTS mode for exposure images, as you already do ---
    orig_unit_fn = pt._data_unit_kind_and_bunit

    def _counts_override():
        kind, bunit = orig_unit_fn()
        if kind == "ambiguous":
            return "counts", bunit
        return kind, bunit

    pt._data_unit_kind_and_bunit = _counts_override

    # --- UI: show exposure, select trail, perform photometry ---
    ui = UI(hduw)
    ui.ax.set_title(f"{target} | obs={observation_id} | {fits_name}")
    ui.add_markers(
        ra1=ra1,
        dec1=dec1,
        ra2=ra2,
        dec2=dec2,
    )  # or your equivalent call for pos1/pos2

    selector = TrailSelector(height=5.0, semi_out=5.0, finalize_on_click=False)
    
    try:
        sel = ui.select_trail(selector)

    except Exception as e:
        print(f"[PHOT] WARN: UI selection failed: {e}")
        return

    ap_box, ann_box = _normalize_selection_to_ap_ann(sel)

    if ap_box is None:
        print(f"[PHOT] User escapes selection. Dubious detection-no photometry result after all")
        row: dict[str, Any] = _build_csv_row(
            target_name=target,
            obs_id=observation_id,
            filt=filter,
            fits_name=fits_name,
            result=None,
            selector=None,
        )

        append_row(
            filepath=config['PHOTOMETRY']['FILEPATH'],
            row=row,
        )
        return

    # --- Save PNG screenshot (if PNG_FILEPATH configured) ---
    try:
        png_path = _build_screenshot_path(
            config=config,
            fits_path=fits_path,
            target=target,
        )
        if png_path is not None:
            ui.fig.savefig(png_path, dpi=150, bbox_inches="tight")
            print(f"[PHOT] PNG exported: {png_path}")

    except Exception as e:
        print(f"[PHOT] WARN: PNG export failed: {e}")

    try:
        res: PhotometryResult = pt.perform_trail_photometry(ap_box, ann_box, debug=True)
        # # --- Build CSV row and append it ---
        row: dict[str, Any] = _build_csv_row(
            target_name=target,
            obs_id=observation_id,
            filt=filter,
            fits_name=fits_name,
            result=res,
            selector=selector,
        )

        append_row(
            filepath=config['PHOTOMETRY']['FILEPATH'],
            row=row,
        )
        
    except Exception as e:
        print(f"[PHOT] WARN: photometry failed: {e}")
        return

