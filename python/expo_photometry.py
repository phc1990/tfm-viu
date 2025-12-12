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

import argparse
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
from configparser import ConfigParser

# Project imports — same style as in photometry.py
from src.photometry.hdu import HDUW
from src.photometry.ui import UI, TrailSelector
from src.photometry.phot import PhotTable

# XSA helpers for source list / OBSMLI
from download import ensure_om_srclist_or_obsmlist

from configparser import ConfigParser  # add this near the imports at the top

# Try to reuse helpers from photometry.py (download root, WCS, PNG, CSV, etc.)
try:
    from photometry import (
        extract_wcs_from_hduw,
        _resolve_download_root_from_ini,
        _dest_dir_for,
        _png_path_for,
        _plot_pos1_pos2,
        _append_csv_row,                  # from photometry.py
        _normalize_selection_to_ap_ann,
        resolve_photometry_csv_from_ini,  # NEW: reuse INI → CSV resolver
    )
    HAVE_PHOTOMETRY_HELPERS = True

except Exception:
    HAVE_PHOTOMETRY_HELPERS = False
    # --- Fallback local implementations (used only if import above fails) ---



def _append_csv_row(csv_path, cols, row, include_headers=True):
        """
        Fallback CSV appender, used ONLY if photometry.py is not importable.
        Writes to the given csv_path like photometry._append_csv_row.
        """
        import csv
        from pathlib import Path
   
        # cols = [
        #     "target_name", "observation_id", "filter", "fits_name",
        #     "mag_ab", "mag_err",
        #     "count_rate", "count_rate_err",
        #     "net_counts", "net_counts_error",
        #     "aper_counts", "bkg_per_pix","bkg_rms_per_pix", "A_ap_eff","A_bg_eff",
        #     "zp_ab",
        #     "trail_height_pix","trail_semi_out_pix","trail_semi_in_pix"
        # ]

        p = Path(csv_path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        write_header = not p.exists() or p.stat().st_size == 0

        row_out = {k: ("" if row.get(k) is None else row.get(k)) for k in cols}

        with p.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            if write_header and include_headers:
                writer.writeheader()
            writer.writerow(row_out)

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

def ensure_photometry_csv_headers(csv_path, cols, include_headers=True):
    """Create CSV early so Ctrl+C does not lose results."""
    csv_path = Path(csv_path).expanduser()
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # cols = [
    #     "target_name","fits_file","observation_id",
    #     "mag_ab","mag_err","count_rate","count_rate_err",
    #     "mode","filter","zp","zp_keyword","zp_source",
    # ]

    if not include_headers:
        if not csv_path.exists():
            csv_path.touch()
        return

    if csv_path.exists() and csv_path.stat().st_size > 0:
        return

    import csv
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()

# -----------------------------------------------------------------------------
def zp_from_ini_for_filter(ini_path: Optional[str], filt: str):
    """
    If screening.ini defines [PHOTOMETRY_RESULTS] ABM0<BAND> for this filter,
    return (zp, keyword). Otherwise (None, None).

    Example:
      filter L  → band UVW1 → key ABM0UVW1
    """
    if not ini_path:
        return None, None

    ini = Path(ini_path).expanduser()
    if not ini.exists():
        return None, None

    cfg = ConfigParser()
    cfg.read(ini)

    if not cfg.has_section("PHOTOMETRY_RESULTS"):
        return None, None

    # Use the same mapping as PhotTable: L→UVW1, M→UVM2, S→UVW2, etc.
    band = PhotTable._filter_to_band(filt)
    if not band:
        return None, None

    key = f"ABM0{band}"
    if not cfg.has_option("PHOTOMETRY_RESULTS", key):
        return None, None

    raw = cfg.get("PHOTOMETRY_RESULTS", key)
    try:
        zp = float(raw)
        return zp, key
    except Exception:
        print(
            f"[EXPO_PHOT] WARN: Could not parse float ZP from "
            f"[PHOTOMETRY_RESULTS] {key}={raw!r}; falling back to OBSMLI."
        )
        return None, None


def parse_args():
    p = argparse.ArgumentParser(
        description="Trail photometry on a single exposure (EXPTIME-based)."
    )
    p.add_argument("--fits", type=str, required=True,
                   help="Exposure FITS file selected in screening.py")
    p.add_argument("--target-name", type=str, required=True)
    p.add_argument("--observation-id", type=str, required=True)
    p.add_argument("--pos1-ra", type=float, required=True)
    p.add_argument("--pos1-dec", type=float, required=True)
    p.add_argument("--pos2-ra", type=float, required=True)
    p.add_argument("--pos2-dec", type=float, required=True)
    p.add_argument("--filter", type=str, required=True)
    p.add_argument("--ini", type=str, required=False,
                   help="Optional screening.ini path (for DOWNLOAD_DIRECTORY, etc.)")
    return p.parse_args()


def _build_csv_row(
    target_name: str,
    obs_id: str,
    filt: str,
    fits_name: str,
    mode: str,
    result,
    selector: TrailSelector,
    pt: PhotTable,
) -> Dict[str, Any]:
    """
    Build a CSV row with both photometric and geometric information.

    This intentionally includes *both* count-rate and counts fields so that:
    - mosaics / rate images remain usable, and
    - for exposures you can focus on net_counts / net_counts_err as requested.
    """

    row: Dict[str, Any] = {
        "target_name": target_name,
        "observation_id": obs_id,
        "filter": filt,
        "mode": mode,
        "fits_name": fits_name,
        # Magnitudes
        "mag_ab": result.mag_ab,
        "mag_err": result.mag_err,
        # Rates (for rate images / mosaics; may be NaN/None for counts-only images)
        "count_rate": result.count_rate,
        "count_rate_err": result.count_rate_err,
        # Counts (this is what you requested for the exposure scenario)
        "net_counts": result.net_counts,
        "net_counts_err": result.net_counts_err,
        "aper_counts": result.aper_counts,
        # Background & areas
        "bkg_per_pix": result.bkg_per_pix,
        "bkg_rms_per_pix": result.bkg_rms_per_pix,
        "A_ap_eff": result.A_ap_eff,
        "A_bg_eff": result.A_bg_eff,
        # Zero-point provenance
        "zp_ab": result.zp_ab,
        "zp_keyword": result.zp_keyword,
        "zp_source_file": result.zp_source_file,
        "zp_source_kind": result.zp_source_kind,
        # Geometry of the trail aperture (if available on the selector)
        "trail_height_pix": getattr(selector, "height", None),
        "trail_semi_out_pix": getattr(selector, "semi_out", None),
        "trail_semi_in_pix": getattr(selector, "semi_in", None),
    }
    return row


# def _main_impl():
#     args = parse_args()

#     fits_path = Path(args.fits)
#     if not fits_path.exists():
#         raise FileNotFoundError(f"Exposure FITS not found: {fits_path}")

#     target = getattr(args, "target_name", None) or "unknown"
#     obs_id = str(getattr(args, "observation_id", "") or "")
#     filt = (getattr(args, "filter", "") or "").upper()

#     ini_arg = getattr(args, "ini", None)
#     ini_path = Path(ini_arg).expanduser() if ini_arg else None

#     # Where to store/download ancillary products (OBSMLI, etc.)
#     dest_root = _resolve_download_root_from_ini(ini_arg)
#     dest_dir = _dest_dir_for(obs_id, filt, dest_root)

#     # First try to get the zero point directly from screening.ini
#     zp_ini, zp_keyword = zp_from_ini_for_filter(ini_arg, filt)

#     # Only download / ensure the OM source list (OBSMLI or equivalent) for ZP
#     # if we do NOT have a ZP from the INI.
#     srclist_path = None
#     srckind = None
#     if zp_ini is None:
#         srclist_path, srckind = ensure_om_srclist_or_obsmlist(
#             str(obs_id),
#             force=False,
#             dest_root=dest_dir,
#         )

#     # No mosaic sky image at all here: we always operate on the *exposure*.
#     used_mosaic = False
#     chosen_fits = fits_path
#     fits_name = chosen_fits.name
#     mode = "EXPOSURE"

#     # Open HDU wrapper and image array
#     hduw = HDUW(file=str(chosen_fits))
#     np.seterr(invalid="ignore")  # avoid float cast warnings
#     arr = np.asarray(
#         getattr(hduw, "data", getattr(hduw, "hdu", None) and getattr(hduw.hdu, "data", None)),
#         dtype=float,
#     )
#     if arr is None or arr.ndim != 2:
#         raise ValueError("expo_photometry: expected a 2-D image in the HDU.")

#     arr = np.ma.masked_invalid(arr)

#     pt = PhotTable(hduw)

#     if zp_ini is not None:
#         # ZP from screening.ini → no OBSMLI needed
#         pt.zero_point = zp_ini
#         # Keep provenance consistent with photometry.py, if available
#         if hasattr(pt, "_zp_prov") and isinstance(getattr(pt, "_zp_prov"), dict):
#             pt._zp_prov.update(
#                 keyword=zp_keyword,
#                 file=str(ini_path) if ini_path is not None else None,
#                 kind="INI",
#             )
#         ini_name = ini_path.name if ini_path is not None else "INI"
#         print(
#             f"[EXPO_PHOT] ZP={pt.zero_point:.6f} from {ini_name} "
#             f"key={zp_keyword} [INI]"
#         )
#     elif srclist_path is not None:
#         # Fallback: original behaviour using OBSMLI / SRCLIST
#         pt.calibrate_against_source_list(
#             source_list_file=str(srclist_path),
#             filter=filt,
#             source_list_kind=str(srckind),
#         )
#     else:
#         print(
#             "[EXPO_PHOT] WARN: No ZP from INI and no SRCLIST file; "
#             "zero_point remains unset."
#         )

#     # --- For EXPOSURES: force COUNTS mode if units are ambiguous ---
#     # For OM FSIMAG exposures, data are in COUNTS even if BUNIT is missing.
#     # phot.py defaults ambiguous+OBSMLI → 'rate', which is wrong here.
#     orig_unit_fn = pt._data_unit_kind_and_bunit


#     def _forced_counts_unit_kind():
#         kind, bunit = orig_unit_fn()
#         if kind == "unknown":
#             # Treat OM exposures as COUNTS images and use EXPTIME
#             return "counts", bunit
#         return kind, bunit

#     pt._data_unit_kind_and_bunit = _forced_counts_unit_kind  # monkey-patch for this instance

    

#     # UI: same behaviour as in photometry.py, but on the exposure
#     ui = UI(hduw)
#     try:
#         if HAVE_PHOTOMETRY_HELPERS:
#             from photometry import _title_for  # type: ignore

#             title = _title_for(hduw, target=target, obs_id=obs_id, filt=filt, pt=pt)
#         else:
#             title = f"{target} | obs={obs_id} | filter={filt} | EXPO"
#         ui.update(title=title)
#     except Exception:
#         pass

#     # Positions from screening.py
#     pos1 = (float(args.pos1_ra), float(args.pos1_dec))
#     pos2 = (float(args.pos2_ra), float(args.pos2_dec))

#     w = extract_wcs_from_hduw(hduw)
#     if w is None:
#         print(
#             "[EXPO_PHOT][WCS] WARN: No celestial WCS found; "
#             "pos1/pos2 will be plotted only if UI accepts raw pixels."
#         )

#     # Optional helper to draw pos1 / pos2 as in photometry.py
#     _plot_pos1_pos2(ui, w, pos1, pos2)

#     # Let the UI know about the endpoints (for markers, etc.)
#     ui.add_sources(pos1, pos2, wcs=w)

#     # Same trail selector as in photometry.py
#     selector = TrailSelector(height=5.0, semi_out=5.0, finalize_on_click=False)
#     sel = ui.select_trail(selector)

#     # Match photometry.py behavior exactly:
#     ap_box, ann_box = _normalize_selection_to_ap_ann(sel)
#     if ap_box is None:
#         print("[EXPO_PHOT] No aperture selected; no photometry performed.")
#         return

#     # Perform photometry
#     result = pt.perform_trail_photometry(
#         rectangular_aperture=ap_box,
#         rectangular_annulus=ann_box,
#         debug=True,
#     )

#     # --- Export PNG of the UI with the aperture and annulus overplotted ---
#     try:
#         png_path = _png_path_for(chosen_fits, target)
#         try:
#             # If your UI exposes a helper like this, use it:
#             if hasattr(ui, "save_png"):
#                 ui.save_png(str(png_path))
#             else:
#                 # Fallback: try saving the matplotlib figure directly
#                 fig = getattr(ui, "fig", None)
#                 if fig is not None:
#                     fig.savefig(str(png_path), dpi=150)
#         except Exception as e:
#             print(f"[EXPO_PHOT] WARN: failed to save PNG ({png_path}): {e}")
#     except Exception as e:
#         print(f"[EXPO_PHOT] WARN: PNG path could not be constructed: {e}")

#     # --- Build CSV row and append (counts + counts_err included) ---
#     row = _build_csv_row(
#         target_name=target,
#         obs_id=obs_id,
#         filt=filt,
#         fits_name=fits_name,
#         mode=mode,
#         result=result,
#         selector=selector,
#         pt=pt,
#     )

#     _append_csv_row(row)

#     # Mirror the MOSAIC summary line from photometry.py, but label as EXPO
#     print(
#         "[PHOT] EXPO: "
#         f"mag={result.mag_ab}  "
#         f"mag_err={result.mag_err}  "
#         f"rate={result.count_rate}  "
#         f"rate_err={result.count_rate_err}  "
#         f"file={fits_name}"
#     )

#     print("[EXPO_PHOT] Photometry row written.")
#     print(
#         f"  mag_AB      = {result.mag_ab:.4f} ± "
#         f"{result.mag_err if result.mag_err is not None else float('nan'):.4f}"
#     )
#     if result.net_counts is not None:
#         print(
#             f"  net_counts  = {result.net_counts:.3f} ± "
#             f"{result.net_counts_err if result.net_counts_err is not None else float('nan'):.3f}"
#         )
#     if result.count_rate is not None:
#         print(
#             f"  count_rate  = {result.count_rate:.5g} ± "
#             f"{result.count_rate_err if result.count_rate_err is not None else float('nan'):.5g}"
#         )

def _main_impl():

    args = parse_args()

    fits_path = Path(args.fits)
    if not fits_path.exists():
        raise FileNotFoundError(f"Exposure FITS not found: {fits_path}")

    target = getattr(args, "target_name", None) or "unknown"
    obs_id = str(getattr(args, "observation_id", "") or "")
    filt = (getattr(args, "filter", "") or "").upper()

    ini_arg = getattr(args, "ini", None)
    ini_path = Path(ini_arg).expanduser() if ini_arg else None



    # --- Resolve CSV path from [PHOTOMETRY_RESULTS] in screening.ini ---
    csv_path, include_headers = resolve_photometry_csv_from_ini(ini_arg)
    if csv_path is None:
        csv_path = Path.home() / "Documents" / "expo_photometry_results.csv"
        print(f"[EXPO_PHOT] Using fallback CSV: {csv_path}")

    # 🔹 NEW: create the CSV file (and headers) immediately, before doing anything else
    cols = [
    "target_name", "observation_id", "filter", "fits_name",
    "mag_ab", "mag_err",
    "count_rate", "count_rate_err",
    "net_counts", "net_counts_error",
    "aper_counts", "bkg_per_pix","bkg_rms_per_pix", "A_ap_eff","A_bg_eff",
    "zp_ab",
    "trail_height_pix","trail_semi_out_pix","trail_semi_in_pix"
    ]

    ensure_photometry_csv_headers(csv_path, cols, include_headers=include_headers)

    # Where to store/download ancillary products (OBSMLI, etc.)
    dest_root = _resolve_download_root_from_ini(ini_arg)
    dest_dir = _dest_dir_for(obs_id, filt, dest_root)

# def _main_impl():
#     args = parse_args()

#     fits_path = Path(args.fits)
#     if not fits_path.exists():
#         raise FileNotFoundError(f"Exposure FITS not found: {fits_path}")

#     target = getattr(args, "target_name", None) or "unknown"
#     obs_id = str(getattr(args, "observation_id", "") or "")
#     filt = (getattr(args, "filter", "") or "").upper()

#     ini_arg = getattr(args, "ini", None)
#     ini_path = Path(ini_arg).expanduser() if ini_arg else None

#     # --- NEW: resolve CSV path via the same logic as photometry.py ---
#     csv_path, include_headers = resolve_photometry_csv_from_ini(ini_arg)
#     if csv_path is None:
#         from pathlib import Path
#         csv_path = Path("expo_photometry.csv")
#         include_headers = True
#         print(f"[EXPO_PHOT] Using fallback CSV: {csv_path}")

#     # Ensure the CSV file exists (header written if requested)
#     ensure_photometry_csv_headers(csv_path, include_headers=include_headers)

    # # Where to store/download ancillary products (OBSMLI, etc.)
    # dest_root = _resolve_download_root_from_ini(ini_arg)
    # dest_dir = _dest_dir_for(obs_id, filt, dest_root)

    # (rest of your ZP / OBSMLI / HDUW / UI / trail selection logic goes here)
    # ZP from INI if available (previous step you added)
    zp_ini, zp_keyword = zp_from_ini_for_filter(ini_arg, filt)

    srclist_path = None
    srckind = None
    if zp_ini is None:
        srclist_path, srckind = ensure_om_srclist_or_obsmlist(
            str(obs_id),
            force=False,
            dest_root=dest_dir,
        )

    # No mosaic sky image here: always operate on the exposure
    used_mosaic = False
    chosen_fits = fits_path
    fits_name = chosen_fits.name
    mode = "EXPOSURE"
    
    # --- Build HDUW / PhotTable, set zero point ---
    hduw = HDUW(chosen_fits)
    hduw = HDUW(file=str(chosen_fits))
    arr = np.asarray(
        getattr(hduw, "data", getattr(hduw.hdu, "data", None)),
        dtype=float
    )
    if arr is None or arr.ndim != 2:
        raise ValueError("UI: expected a 2-D image in the HDU.")

    arr = np.ma.masked_invalid(arr)  # what you’re displaying

    pt = PhotTable(hduw)

    if zp_ini is not None:
        pt.zero_point = zp_ini
        if hasattr(pt, "_zp_prov") and isinstance(getattr(pt, "_zp_prov"), dict):
            pt._zp_prov.update(
                keyword=zp_keyword,
                file=str(ini_path) if ini_path is not None else None,
                kind="INI",
            )
        ini_name = ini_path.name if ini_path is not None else "INI"
        print(
            f"[EXPO_PHOT] ZP={pt.zero_point:.6f} from {ini_name} key={zp_keyword} [INI]"
        )
    elif srclist_path is not None:
        pt.calibrate_against_source_list(
            source_list_file=str(srclist_path),
            filter=filt,
            source_list_kind=str(srckind),
        )
    else:
        print(
            "[EXPO_PHOT] WARN: No ZP from INI and no SRCLIST file; "
            "zero_point remains unset."
        )

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
    #ui.ax.set_title(f"{target} | obs={obs_id} | {fits_name} | {mode} | filter={filt}")
    ui.ax.set_title(f"{target} | obs={obs_id} | {fits_name}")
    ui.add_markers_from_args(args)  # or your equivalent call for pos1/pos2

    # selector = TrailSelector(ui=ui, pt=pt)
    # trail = selector.ask_trail()
    
    selector = TrailSelector(height=5.0, semi_out=5.0, finalize_on_click=False)
    
    try:
        sel = ui.select_trail(selector)

    except Exception as e:
        print(f"[PHOT] WARN: UI selection failed: {e}")
        _append_csv_row(csv_path, cols, target=target, obs_id=obs_id, fits_file=fits_name, filt=filt,
                      mode="EXPOSURE",
                      reason="ui_error")
        return

    ap_box, ann_box = _normalize_selection_to_ap_ann(sel)


    if ap_box is None:
        _append_csv_row(csv_path, cols, target=target, obs_id=obs_id, fits_file=fits_name, filt=filt,
                      mode="EXPOSURE",
                      reason="no_selection")
        return


    # result = selector.perform_trail_photometry(trail)
    # ui.close()

    # --- Save PNG screenshot (if PNG_FILEPATH configured) ---
    try:
        png_path = _png_path_for(chosen_fits, target, ini_path)
        if png_path is not None:
            ui.fig.savefig(png_path, dpi=150, bbox_inches="tight")
            print(f"[PHOT] PNG exported: {png_path}")

    except Exception as e:
        print(f"[PHOT] WARN: PNG export failed: {e}")

    try:
        res: PhotometryResult = pt.perform_trail_photometry(ap_box, ann_box, debug=True)
        # # --- Build CSV row and append it ---
        row = _build_csv_row(
            target_name=target,
            obs_id=obs_id,
            filt=filt,
            fits_name=fits_name,
            mode=mode,
            result=res,
            selector=selector,
            pt=pt,
        )

        _append_csv_row(csv_path, cols, row, include_headers=include_headers)
        
    except Exception as e:
        print(f"[PHOT] WARN: photometry failed: {e}")
        _append_csv_row(csv_path, cols, target=target, obs_id=obs_id, fits_file=fits_name, filt=filt,
                      mode="CURRENT", reason=str(e))
        return


def main():
    try:
        _main_impl()
    except KeyboardInterrupt:
        # User cancelled during the UI / selection stage
        print("\n[EXPO_PHOT] Aborted by user. No new photometry row written.")
        # Any existing CSV rows from previous runs are already on disk.
        return

if __name__ == "__main__":
    main()
