#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
from configparser import ConfigParser
from pathlib import Path
from typing import Any, Optional, Tuple
import re
from urllib.parse import urlencode
import numpy as np
from photutils.aperture import RectangularAnnulus, RectangularAperture
import subprocess
from src.photometry.hdu import HDUW
from src.photometry.phot import (
    PhotTable,
    PhotometryResult,
    AB_PROP_COEFF,
    arcsec_to_pix, 
    compute_c1_for_star
)
from src.photometry.ui import UI, TrailSelector
from src.screening.xsa import convert_filter_name_to_xsa_name
from astropy.io import fits as _fits
from src.photometry.utils import angle_to_rad

from common import (
    OBS_ID_COLS,
    TARGET_COLS,
    FILTER_COLS,
    FITS_FILE_COLS,
    POS1_DEC_COLS,
    POS1_RA_COLS,
    POS2_DEC_COLS,
    POS2_RA_COLS,
    V_MAG_1_COL,
    V_MAG_1_CORRECTED_COL,
    MLIM_OBS_COL,
    extract_row_value,
    append_row,
)

# ---------------------------------------------------------------------
# Helpers from photometry.py (normalization of UI selection to apertures)
# ---------------------------------------------------------------------
try:
    from photometry import _normalize_selection_to_ap_ann  # type: ignore
    HAVE_PHOTOMETRY_HELPERS = True
except Exception:
    HAVE_PHOTOMETRY_HELPERS = False


# ---------------------------------------------------------------------
# CSV header migration (required because common.append_row rejects unknown keys)
# ---------------------------------------------------------------------
def ensure_csv_has_fields(csv_path: str | Path, required_fields: list[str]) -> None:
    """
    Ensure a CSV exists and contains all required_fields in its header.
    If the CSV exists but is missing columns, rewrite it with an extended header.
    """
    p = Path(csv_path)
    required_fields = list(required_fields)

    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=required_fields)
            w.writeheader()
        return

    with p.open("r", newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r, None)

    if not header:
        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=required_fields)
            w.writeheader()
        return

    missing = [c for c in required_fields if c not in header]
    if not missing:
        return

    new_header = header + missing

    rows: list[dict[str, Any]] = []
    with p.open("r", newline="", encoding="utf-8") as f:
        dr = csv.DictReader(f)
        for row in dr:
            for c in missing:
                row.setdefault(c, "")
            rows.append(row)

    with p.open("w", newline="", encoding="utf-8") as f:
        dw = csv.DictWriter(f, fieldnames=new_header)
        dw.writeheader()
        for row in rows:
            dw.writerow(row)


# ---------------------------------------------------------------------
# Deterministic SRCLIST finder (per exposure)
# ---------------------------------------------------------------------
def find_srclist_in_same_dir(fits_path: Path) -> Path | None:
    """
    Locate the SWSRLI FTZ matching this exposure FITS.

    Uses exposure prefix 'P<obsid>OMS###' from the exposure filename and finds
    a file with 'SWSRLI' that starts with the same prefix in the same folder.

    Example:
      exposure: P0821871601OMS006FSIMAGL000.FTZ
      srclist:  P0821871601OMS006SWSRLIL000.FTZ
    """
    fits_path = Path(fits_path)
    folder = fits_path.parent
    if not folder.exists() or not folder.is_dir():
        return None

    name_u = fits_path.name.upper()
    idx = name_u.find("OMS")
    if idx < 0 or len(name_u) < idx + 6:
        return None
    prefix = name_u[: idx + 6]  # P<obsid>OMS###

    for p in folder.iterdir():
        if not p.is_file():
            continue
        pn = p.name.upper()
        if pn.startswith(prefix) and "SWSRLI" in pn and pn.endswith(".FTZ"):
            return p
    return None

# ---------------------------------------------------------------------
# Helpers to download SRCLIST per exposure
# ---------------------------------------------------------------------

def download_srclist_ftz(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    args = [
        "curl", "-L", "-f", "-sS",
        "--connect-timeout", "15",
        "--max-time", "180",
        "--retry", "3",
        "--retry-delay", "2",
        "-o", str(out_path),
        url,
    ]
    res = subprocess.run(args, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(res.stderr.strip())

    # basic sanity check
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError("downloaded SRCLIST is empty or missing")




def build_srclist_url(base: str, obsno: str, filt: str, expno: str) -> str:
    params = {
        "obsno": obsno,
        "instname": "OM",
        "level": "PPS",
        "name": "SWSRLI",
        "extension": "FTZ",
        "filter": filt,
        "expno": expno,
    }
    return base.rstrip("?") + "?" + urlencode(params)

def expno_from_filename(fits_name: str) -> str | None:
    m = re.search(r"OMS(\d{3})", fits_name)
    return m.group(1) if m else None


# ---------------------------------------------------------------------
# ZP helper
# ---------------------------------------------------------------------
def zp_from_ini_for_filter(config: ConfigParser, filt: str) -> Optional[float]:
    """
    Return AB zero point for this filter from screening.ini [PHOTOMETRY], if present.
    Keys are ABM0<BAND> where BAND is PhotTable band name (L→UVW1, M→UVM2, ...).
    """
    band = PhotTable.om_filter_to_band(filt)
    if not band:
        return None
    key = f"ABM0{band}"
    raw = config.get("PHOTOMETRY", key, fallback=None)
    if raw in (None, "", "None"):
        return None
    try:
        return float(raw)
    except Exception:
        return None


# ---------------------------------------------------------------------
# Coincidence-loss (CoI) helper
# ---------------------------------------------------------------------
def _parse_float_list(raw: str | None) -> list[float]:
    if raw in (None, "", "None"):
        return []
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    out: list[float] = []
    for p in parts:
        try:
            out.append(float(p))
        except Exception:
            continue
    return out


def coi_f_polynomial(x: float, coeffs: list[float]) -> float:
    """Evaluate polynomial f(x) = a0 + a1 x + a2 x^2 + ...

    If coeffs is empty, returns 1.
    """
    if not coeffs:
        return 1.0
    y = 0.0
    xp = 1.0
    for a in coeffs:
        y += float(a) * xp
        xp *= float(x)
    return float(y)


def apply_coi_correction(
    R_raw: float,
    frame_time: float,
    dead_fraction: float,
    f_coeffs: list[float] | None = None,
) -> float:
    """Apply OM coincidence-loss correction to a total (source+background) rate.

    Implements:
      R_corr = - ln(1 - R_raw * t_f) / (t_f * (1 - d_f)) * f(R_raw * t_f)

    Notes
    -----
    * R_raw must be the TOTAL rate in the CoI aperture (source+background).
    * f(x) is an empirical polynomial term; if not provided, f(x)=1.
    """
    tf = float(frame_time)
    df = float(dead_fraction)
    if not np.isfinite(R_raw) or not np.isfinite(tf) or not np.isfinite(df):
        raise ValueError("CoI: non-finite inputs")
    if tf <= 0:
        raise ValueError("CoI: invalid frame_time")
    if not (0.0 <= df < 1.0):
        raise ValueError("CoI: invalid dead fraction")

    x = float(R_raw) * tf
    # Numerical safety: x must be < 1
    if x >= 1.0:
        raise ValueError("CoI: R_raw * frame_time >= 1 (saturation)")
    if x <= 0.0:
        return float(R_raw)

    base = -np.log(1.0 - x) / (tf * (1.0 - df))
    f = coi_f_polynomial(x, f_coeffs or [])
    return float(base * f)


def coi_derivative_dR(
    R_raw: float,
    frame_time: float,
    dead_fraction: float,
    f_coeffs: list[float] | None = None,
) -> float:
    """dR_corr/dR_raw for uncertainty propagation.
    For simplicity we ignore d f(x)/dx for the polynomial term since it is set to f(x) ==1 (ask simon)
    """
    tf = float(frame_time)
    df = float(dead_fraction)
    x = float(R_raw) * tf
    if x >= 1.0:
        return float("nan")
    denom = (1.0 - x) * (1.0 - df)
    if denom <= 0:
        return float("nan")
    f = coi_f_polynomial(x, f_coeffs or [])
    return float(f / denom)


# ---------------------------------------------------------------------
# Screenshot path helper
# ---------------------------------------------------------------------
def _build_screenshot_path(config: ConfigParser, fits_path: Path, target: str) -> Path:
    base_dir = Path(config["PHOTOMETRY"]["SCREENSHOOTS_DIRECTORY"]).expanduser()
    base_dir.mkdir(parents=True, exist_ok=True)

    target_safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in (target or "")) or "target"
    base_name = f"{fits_path.stem}__{target_safe}"
    candidate = base_dir / f"{base_name}.png"

    i = 1
    while candidate.exists():
        candidate = base_dir / f"{base_name}__{i}.png"
        i += 1
    return candidate



# ---------------------------------------------------------------------
# Robust scatter (MAD)
# ---------------------------------------------------------------------
def _mad_sigma(x: list[float]) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return 0.0
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(1.4826 * mad)

# def _angle_to_rad(theta: Any) -> float:
#     """
#     Return theta as a plain float in radians.

#     photutils/astropy apertures may store theta either as:
#       - a plain float
#       - an astropy Quantity with angular units, usually rad
#     """
#     if theta is None:
#         return 0.0

#     try:
#         # astropy Quantity case
#         if hasattr(theta, "to_value"):
#             import astropy.units as u
#             return float(theta.to_value(u.rad))
#     except Exception:
#         pass

#     try:
#         return float(theta)
#     except Exception:
#         return 0.0


# ---------------------------------------------------------------------
# Build photometry CSV row (includes apcorr columns always)
# ---------------------------------------------------------------------
def _build_csv_row(
    target_name: str,
    obs_id: str,
    filt: str,
    fits_name: str,
    result: Optional[PhotometryResult],
    selector: Optional[TrailSelector],
) -> dict[str, Any]:
    row: dict[str, Any] = {
    # Identification
    "target_name": target_name,
    "observation_id": obs_id,
    "filter": filt,
    "fits_name": fits_name,

    # Final science products
    "mag_ab": None,
    "mag_err": None,
    "mag_ab_apcorr": None,
    "mag_ab_apcorr_err": None,

    # Raw trail photometry
    "count_rate": None,
    "count_rate_err": None,
    "net_counts": None,
    "net_counts_err": None,
    "aper_counts": None,
    "bkg_per_pix": None,
    "bkg_rms_per_pix": None,
    "bkg_counts_ap": None,
    "bkg_counts_ap_err": None,
    "A_ap_eff": None,
    "A_bg_eff": None,
    "zp_ab": None,

    # Geometry
    "trail_height_pix": None,
    "trail_semi_out_pix": None,
    "trail_semi_in_pix": None,

    # Correction factors
    "tds_corr": None,
    "c2": None,
    "c2_err": None,

    "f_apcorr": None,
    "f_apcorr_err": None,
    "apcorr_mag": None,
    "apcorr_mag_err": None,

    "c1": None,
    "c1_err": None,
    "c1_detail_csv": None,
    "coi_factor": None,

    # Rate-domain correction chain
    "rate_bkg_6": None,
    "rate_bkg_6_coi": None,
    "rate_tot_6": None,
    "rate_tot_6_coi": None,

    "count_rate_coi": None,
    "count_rate_coi_err": None,

    "count_rate_final": None,
    "count_rate_final_err": None,

    # Diagnostic APCORR branch only
    "count_rate_final_apcorr": None,
    "count_rate_final_apcorr_err": None,

    # Screening/catalogue context
    "v_mag_1": None,
    "v_mag_1_corrected": None,
    "mlim_obs": None,
}

    if result is None:
        return row
    
    bkg_counts_ap = None
    bkg_counts_ap_err = None

    if (result.bkg_per_pix is not None) and (result.A_ap_eff is not None):
        bkg_counts_ap = float(result.bkg_per_pix) * float(result.A_ap_eff)

    if (
        result.bkg_rms_per_pix is not None
        and result.A_ap_eff is not None
        and result.A_bg_eff is not None
        and np.isfinite(float(result.A_bg_eff))
        and float(result.A_bg_eff) > 0
    ):
        bkg_counts_ap_err = (
            float(result.A_ap_eff)
            * float(result.bkg_rms_per_pix)
            / np.sqrt(float(result.A_bg_eff))
        )

    row.update(
        {
            "mag_ab": result.mag_ab,
            "mag_err": result.mag_err,
            "count_rate": result.count_rate,
            "count_rate_err": result.count_rate_err,
            "net_counts": result.net_counts,
            "net_counts_err": result.net_counts_err,
            "aper_counts": result.aper_counts,
            "bkg_per_pix": result.bkg_per_pix,
            "bkg_rms_per_pix": result.bkg_rms_per_pix,
            "bkg_counts_ap": bkg_counts_ap,
            "bkg_counts_ap_err": bkg_counts_ap_err,
            "A_ap_eff": result.A_ap_eff,
            "A_bg_eff": result.A_bg_eff,
            "zp_ab": result.zp_ab,
        }
    )
    return row


# ---------------------------------------------------------------------
# Write apcorr star CSV into APCORR_DIRECTORY and compute median apcorr
# ---------------------------------------------------------------------
def _write_apcorr_star_csv(
    *,
    ui: UI,
    pt: PhotTable,
    srclist_path: Path,
    apcorr_dir: Path,
    fits_name: str,
    observation_id: str,
    filt: str,
    target: str,
) -> Tuple[int, Path, Optional[Tuple[float, float]]]:
    star_sel = getattr(ui, "calib_star_selections", {}) or {}

    # Requested: store under [PHOTOMETRY] APCORR_DIRECTORY
    apcorr_dir = Path(apcorr_dir).expanduser()
    apcorr_dir.mkdir(parents=True, exist_ok=True)

    # Keep SRCLIST filename to avoid collisions; add .csv
    out_csv = apcorr_dir / (Path(srclist_path).name + ".csv")

    if not star_sel:
        print("[PHOT] NOTE: no calibration stars selected; apcorr CSV not written.")
        return 0, out_csv, None

    rate_arr = getattr(ui, "_srclist_rate", None)
    rateerr_arr = getattr(ui, "_srclist_rate_err", None)

    if rate_arr is None:
        print("[PHOT] WARN: ui._srclist_rate is None; cannot compute apcorr from SRCLIST.")
        return 0, out_csv, None

    header = [
        "slot", "srclist_index",
        "x", "y", "width", "height", "semi_out", "theta",
        "rate_srclist", "rate_err_srclist",
        "count_rate_rect", "count_rate_rect_err",
        "f_apcorr_i", "apcorr_mag_i",
        "fits_name", "srclist_name", "obs_id", "filter", "target",
    ]

    wrote_header = not out_csv.exists()
    apcorr_mags: list[float] = []

    with out_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if wrote_header:
            w.writeheader()

        for slot, d in sorted(star_sel.items()):
            try:
                idx = int(d["srclist_index"])
                cx = float(d["x"])
                cy = float(d["y"])
                width = float(d["width"])
                height = float(d["height"])
                semi_out = float(d["semi_out"])
                theta = float(d.get("theta", 0.0))
            except Exception as e:
                print(f"[PHOT] WARN: invalid calib selection slot A{slot}: {e}")
                continue

            if idx < 0 or idx >= len(rate_arr):
                print(f"[PHOT] WARN: slot A{slot} idx={idx} out of range for SRCLIST arrays; skipping.")
                continue

            r = float(rate_arr[idx])
            rerr = float(rateerr_arr[idx]) if (rateerr_arr is not None and idx < len(rateerr_arr)) else float("nan")

            ap = RectangularAperture((cx, cy), w=width, h=height, theta=theta)
            ann = RectangularAnnulus(
                (cx, cy),
                w_in=width, h_in=height,
                w_out=width + 2.0 * semi_out,
                h_out=height + 2.0 * semi_out,
                theta=theta,
            )

            star_res = pt.perform_trail_photometry(ap, ann, debug=False)
            c = float(star_res.count_rate) if star_res.count_rate is not None else float("nan")
            cerr = float(star_res.count_rate_err) if star_res.count_rate_err is not None else float("nan")

            if not (np.isfinite(r) and np.isfinite(c) and r > 0 and c > 0):
                print(f"[PHOT] WARN: invalid rates for slot A{slot}: RATE={r}, c_rect={c}; skipping.")
                continue

            f_i = r / c
            ap_i = -2.5 * np.log10(f_i)
            apcorr_mags.append(float(ap_i))

            w.writerow(
                {
                    "slot": slot,
                    "srclist_index": idx,
                    "x": cx, "y": cy,
                    "width": width, "height": height,
                    "semi_out": semi_out, "theta": theta,
                    "rate_srclist": r,
                    "rate_err_srclist": rerr,
                    "count_rate_rect": c,
                    "count_rate_rect_err": cerr,
                    "f_apcorr_i": f_i,
                    "apcorr_mag_i": ap_i,
                    "fits_name": fits_name,
                    "srclist_name": Path(srclist_path).name,
                    "obs_id": observation_id,
                    "filter": filt,
                    "target": target,
                }
            )

    if apcorr_mags:
        mags = np.asarray(apcorr_mags, dtype=float)
        apcorr_mag = float(np.median(mags))
        apcorr_mag_err = _mad_sigma(mags)
        print(f"[PHOT] apcorr_mag={apcorr_mag:.5f} mag (N={len(mags)}), scatter={apcorr_mag_err:.5f} mag")
        return len(mags), out_csv, (apcorr_mag, apcorr_mag_err)

    print("[PHOT] WARN: no valid stars measured; apcorr not computed.")
    return 0, out_csv, None

# ---------------------------------------------------------------------
# C1 factor: hbox -> 6 arcsec standard height, only when hbox < h6
# ---------------------------------------------------------------------
def _compute_c1_if_needed(
    *,
    config: ConfigParser,
    hduw: HDUW,
    pt: PhotTable,
    srclist_path: Optional[Path],
    fits_name: str,
    observation_id: str,
    filt: str,
    target: str,
    trail_width_pix: float,
    trail_height_pix: float,
    trail_theta_rad: float,
    trail_semi_out_pix: float,
    height6_pix: float,
) -> tuple[float, float, Optional[Path]]:
    """
    Compute C1 = R_6arcsec / R_hbox only if the asteroid trail box height
    is smaller than the standard 6 arcsec radius equivalent box.

    Returns:
      (C1, C1_err, detail_csv_path)

    If not needed, returns (1.0, 0.0, None).
    If needed but no valid stars are measured, falls back to [PHOTOMETRY] C1.
    """

    # Small tolerance to avoid triggering C1 for 12.6 vs 13 px rounding noise.
    if (
        not np.isfinite(trail_height_pix)
        or not np.isfinite(height6_pix)
        or trail_height_pix <= 0
        or height6_pix <= 0
        or trail_height_pix >= (height6_pix - 0.5)
    ):
        return 1.0, 0.0, None

    c1_fallback = float(config.get("PHOTOMETRY", "C1", fallback="1.0") or 1.0)

    print(
        "[PHOT][C1] Needed: "
        f"trail_height={trail_height_pix:.2f}px < height6={height6_pix:.2f}px. "
        "Opening C1 calibration step."
    )

    # Output directory: reuse APCORR_DIRECTORY unless a C1_DIRECTORY is defined.
    c1_dir_raw = config.get("PHOTOMETRY", "C1_DIRECTORY", fallback="").strip()
    if c1_dir_raw:
        c1_dir = Path(c1_dir_raw).expanduser()
    else:
        apcorr_dir_raw = config.get("PHOTOMETRY", "APCORR_DIRECTORY", fallback="").strip()
        if apcorr_dir_raw:
            c1_dir = Path(apcorr_dir_raw).expanduser().parent / "c1"
        else:
            c1_dir = Path(config["PHOTOMETRY"]["FILEPATH"]).expanduser().parent / "c1"

    c1_dir.mkdir(parents=True, exist_ok=True)

    target_safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in (target or "")) or "target"
    out_csv = c1_dir / f"c1_details_{target_safe}_{Path(fits_name).stem}.csv"

    # New UI on the same image, only for C1 stars.
    ui_c1 = UI(hduw)

    # Reuse the existing C2 overlay machinery:
    # for C1, the "large" overlay is not 35 arcsec, but the standard 6 arcsec box.
    ui_c1 = UI(hduw)

    # ---------------------------------------------------------
    # C1 geometry
    # ---------------------------------------------------------
    c1_width_pix = float(
        arcsec_to_pix(hduw, 12.0)
    )

    c1_bg_semi_out_pix = (
        float(trail_semi_out_pix)
        if (
            np.isfinite(trail_semi_out_pix)
            and trail_semi_out_pix > 0
        )
        else 6.0
    )

    # Initially no extra gap.
    # If visual inspection shows that the local annulus is
    # problematic, this can be increased independently.
    c1_bg_gap_pix = 0.0

    # Tell UI that this is C1, not C2.
    ui_c1.calib_overlay_mode = "c1"

    ui_c1.c1_height6_pix = float(height6_pix)
    ui_c1.c1_bg_semi_out_pix = float(
        c1_bg_semi_out_pix
    )
    ui_c1.c1_bg_gap_pix = float(
        c1_bg_gap_pix
    )

    # Useful for screenshots/debugging: keep saved A# boxes visible by default.
    ui_c1.keep_calib_boxes = True
    c1_width_pix = float(arcsec_to_pix(hduw, 12.0))
    if srclist_path is not None:
        try:
            ui_c1.add_srclist_overlay(srclist_path)
        except Exception as e:
            print(f"[PHOT][C1] WARN: could not overlay SRCLIST: {e}")

    title = (
        f"C1 calibration — {fits_name}\n"
        f"Rh: {c1_width_pix:.1f} x "
        f"{trail_height_pix:.1f}px | "
        f"R6: {c1_width_pix:.1f} x "
        f"{height6_pix:.1f}px | "
        f"common BG annulus"
    )
    try:
        ui_c1.fig.suptitle(title, fontsize=10)
    except Exception:
        ui_c1.ax.set_title(title, fontsize=10)

    # The selector is only used to drive the star-calibration UI.
    # Start with the same hbox height and annulus thickness as the asteroid box
    selector_c1 = TrailSelector(
        height=float(trail_height_pix),
        semi_out=float(c1_bg_semi_out_pix),
        finalize_on_click=False,
    )

    # Fixed C1 length L = 12 arcsec
    selector_c1.width = float(c1_width_pix)

    # Same orientation as asteroid trail
    selector_c1.theta = float(trail_theta_rad)

    try:
        ui_c1.select_trail(selector_c1)
    except Exception as e:
        print(f"[PHOT][C1] WARN: C1 UI failed: {e}")
        return c1_fallback, 0.0, None

    sels: dict[int, dict[str, Any]] = getattr(ui_c1, "calib_star_selections", {}) or {}
    if not sels:
        print(f"[PHOT][C1] WARN: no C1 stars selected; using config C1={c1_fallback:.6f}")
        return c1_fallback, 0.0, None

    header = [
        "slot",
        "x", "y",
        "width_pix",
        "theta_rad",
        "height_h_pix",
        "height6_pix",
        "semi_out_pix",
        "rate_h",
        "rate_h_err",
        "rate6",
        "rate6_err",
        "c1",
        "c1_err",
        "fits_name",
        "obs_id",
        "filter",
        "target",
    ]

    c1_values: list[float] = []

    wrote_header = not out_csv.exists()
    with out_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if wrote_header:
            w.writeheader()

        for slot, d in sorted(sels.items()):
            try:
                x = float(d["x"])
                y = float(d["y"])

                # Prefer the width/theta stored by the UI selection.
                # Fallback to the asteroid trail geometry.
                # width_pix = float(trail_width_pix)
                # NOTE: Simon: fixed stellar-box length L = 12 arcsec. 
                width_pix = float(arcsec_to_pix(hduw, 12.0))
                theta = float(trail_theta_rad)

                # For C1, force the small height to be the asteroid hbox.
                # The standard height is height6_pix.
                meas = compute_c1_for_star(
                    phot=pt,
                    x=x,
                    y=y,
                    width_pix=float(c1_width_pix),
                    theta_rad=float(trail_theta_rad),
                    height_h_pix=float(trail_height_pix),
                    height6_pix=float(height6_pix),
                    semi_out_pix=float(c1_bg_semi_out_pix),
                    bg_center=None,
                    bg_gap_pix=float(c1_bg_gap_pix),
                    debug=False,
                )
                
                meas.slot = int(slot)
                print(
                    f"[PHOT][C1] A{slot}: using asteroid geometry "
                    f"width={width_pix:.2f}px theta={theta:.4f}rad "
                    f"h={trail_height_pix:.2f}px h6={height6_pix:.2f}px"
                )

                if not np.isfinite(meas.c1) or meas.c1 <= 0:
                    print(f"[PHOT][C1] WARN: invalid C1 in slot A{slot}: {meas.c1}")
                    continue

                c1_values.append(float(meas.c1))

                w.writerow(
                    {
                        "slot": meas.slot,
                        "x": x,
                        "y": y,
                        "width_pix": width_pix,
                        "theta_rad": theta,
                        "height_h_pix": trail_height_pix,
                        "height6_pix": height6_pix,
                        "semi_out_pix": trail_semi_out_pix,
                        "rate_h": meas.rate_h,
                        "rate_h_err": meas.err_h,
                        "rate6": meas.rate6,
                        "rate6_err": meas.err6,
                        "c1": meas.c1,
                        "c1_err": meas.c1_err,
                        "fits_name": fits_name,
                        "obs_id": observation_id,
                        "filter": filt,
                        "target": target,
                    }
                )

                print(
                    f"[PHOT][C1] A{meas.slot}: "
                    f"C1={meas.c1:.6f} "
                    f"(R6={meas.rate6:.6g} / Rh={meas.rate_h:.6g})"
                )

            except Exception as e:
                print(f"[PHOT][C1] WARN: slot A{slot} failed: {e}")

    if not c1_values:
        print(f"[PHOT][C1] WARN: no valid C1 measurements; using config C1={c1_fallback:.6f}")
        return c1_fallback, 0.0, out_csv

    c1_arr = np.asarray(c1_values, dtype=float)
    C1 = float(np.median(c1_arr))
    C1_err = _mad_sigma(list(c1_arr))

    print(f"[PHOT][C1] C1={C1:.6f}  scatter={C1_err:.6f}  N={len(c1_arr)}")
    print(f"[PHOT][C1] Details saved: {out_csv}")

    return C1, C1_err, out_csv

# ---------------------------------------------------------------------
# Main callable used by start.py
# ---------------------------------------------------------------------
def action_photometry(config: ConfigParser, screening_row: dict[str, Any]) -> None:
    if not HAVE_PHOTOMETRY_HELPERS:
        raise RuntimeError("photometry._normalize_selection_to_ap_ann could not be imported; cannot proceed.")

    observation_id: str = extract_row_value(screening_row, OBS_ID_COLS)
    target: str = extract_row_value(screening_row, TARGET_COLS)
    filt: str = extract_row_value(screening_row, FILTER_COLS).upper()

    ra1: float = float(extract_row_value(screening_row, POS1_RA_COLS))
    dec1: float = float(extract_row_value(screening_row, POS1_DEC_COLS))
    ra2: float = float(extract_row_value(screening_row, POS2_RA_COLS))
    dec2: float = float(extract_row_value(screening_row, POS2_DEC_COLS))

    fits_name: str = extract_row_value(screening_row, FITS_FILE_COLS)
    fits_path = Path(config["INPUT"]["DOWNLOAD_DIRECTORY"]) / observation_id / filt / fits_name
    if not fits_path.exists():
        raise FileNotFoundError(f"Exposure FITS not found: {fits_path}")

    # ZP from INI (optional)
    zp_ini = zp_from_ini_for_filter(config=config, filt=filt)

    # Build HDUW / PhotTable (no duplicates)
    hduw = HDUW(file=str(fits_path))

    # Sanity-check image shape without casting (avoid dtype warnings)
    raw = getattr(hduw, "data", None)
    if raw is None:
        raw = getattr(getattr(hduw, "hdu", None), "data", None)
    if raw is None or getattr(raw, "ndim", 0) != 2:
        raise ValueError("UI: expected a 2-D image in the HDU.")

    pt = PhotTable(hduw)
    if zp_ini is not None:
        pt.zero_point = zp_ini

    # Force COUNTS mode for exposure images
    orig_unit_fn = pt._data_unit_kind_and_bunit
    def _counts_override():
        kind, bunit = orig_unit_fn()
        if kind == "ambiguous":
            return "counts", bunit
        return kind, bunit
    pt._data_unit_kind_and_bunit = _counts_override

    # UI
    ui = UI(hduw)
    ui.ax.set_title(f"{target} | obs={observation_id} | {fits_name}")

    # WCS (usar el que ya venga en HDUW si existe)
    wcs = getattr(hduw, "wcs", None)
    if wcs is None:
        try:
            from astropy.wcs import WCS
            wcs = WCS(hduw.hdu.header)
        except Exception:
            wcs = None

    # Debug útil (opcional)
    print(f"[PHOT][DBG] ra1/dec1=({ra1},{dec1}) ra2/dec2=({ra2},{dec2}) wcs={'OK' if wcs is not None else 'None'}")

    try:
        ui.add_markers(ra1=ra1, dec1=dec1, ra2=ra2, dec2=dec2, wcs=wcs)
    except Exception:
        pass

    # SRCLIST overlay (optional)
    srclist_path = find_srclist_in_same_dir(fits_path)

    if srclist_path is None:
        print(f"[PHOT] NOTE: SRCLIST not found next to {fits_path.name}; trying download SWSRLI...")

        expno = expno_from_filename(fits_path.name)
        if expno is None:
            print(f"[PHOT] WARN: cannot derive expno from filename {fits_path.name}; skipping SRCLIST download.")
        else:
            try:
                xsa_filt = convert_filter_name_to_xsa_name(filt)
                base = config["INPUT"]["BASE_URL"]
                url = build_srclist_url(base, obsno=observation_id, filt=xsa_filt, expno=expno)
                print(
                    "[PHOT][DBG] SRCLIST AIO params:",
                    f"obsno={observation_id} instname=OM level=PPS name=SWSRLI extension=FTZ filter={filt} expno={expno}"
                )
                print(f"[PHOT][DBG] SRCLIST AIO URL: {url}")

                # choose a deterministic output name
                out_path = fits_path.parent / f"P{observation_id}OMS{expno}FSWSRLI{filt}000.FTZ"
                download_srclist_ftz(url, out_path)
            except Exception as e:
                print(f"[PHOT] WARN: SRCLIST download failed: {e}")

        srclist_path = find_srclist_in_same_dir(fits_path)

    if srclist_path is None:
        print(f"[PHOT] NOTE: SRCLIST still not found; skipping overlay.")
    else:
        ui.add_srclist_overlay(srclist_path)
        
        #Store TDS_CORR keyword from sourcelist
        tds_corr = None
        try:
            with _fits.open(str(srclist_path), memmap=False) as _hdul:
                # tds_corr = float(_hdul[0].header.get("TDS_CORR", 1.0))
                for hdu in _hdul:
                    val = hdu.header.get("TDS_CORR")
                    if val is not None:
                        try:
                            val = float(val)
                            if np.isfinite(val):
                                tds_corr = val
                                break
                        except (TypeError, ValueError):
                            pass
                
                if tds_corr is None:
                    print("[PHOT] WARN: TDS_CORR not found; using 1.0")
                    tds_corr = 1.0

        except Exception:
            tds_corr = 1.0



    # Default trail-box height equivalent to r=6" (full width = 12")
    # 1) Standard 6"-equivalent box height (full width = 12")
    #    We approximate A_6_eff by scaling A_h_eff by the ratio of heights,
    #    keeping the same trail length/width.
    try:

        height6_pix = float(arcsec_to_pix(hduw, 12.0))
    except Exception:
        # fallback: assume binned ~0.95"/pix; 12" ~ 12.6 pix
        height6_pix = 12.6
    # selector = TrailSelector(height=13.0, semi_out=6.0, finalize_on_click=False)
    selector = TrailSelector(height=height6_pix, semi_out=6.0, finalize_on_click=False)
    try:
        sel = ui.select_trail(selector)
    except Exception as e:
        print(f"[PHOT] WARN: UI selection failed: {e}")
        return

    ap_box, ann_box = _normalize_selection_to_ap_ann(sel)

    # Main photometry CSV: migrate header upfront (append_row rejects unknown keys)
    phot_csv = config["PHOTOMETRY"]["FILEPATH"]
    required_fields = list(
        _build_csv_row(
            target_name=target,
            obs_id=observation_id,
            filt=filt,
            fits_name=fits_name,
            result=None,
            selector=selector,
        ).keys()
    )

    for extra_col in ["v_mag_1", "v_mag_1_corrected", "mlim_obs"]:
        if extra_col not in required_fields:
            required_fields.append(extra_col)

    ensure_csv_has_fields(phot_csv, required_fields)
    # ensure_csv_has_fields(
    #     phot_csv,
    #     list(
    #         _build_csv_row(
    #             target_name=target,
    #             obs_id=observation_id,
    #             filt=filt,
    #             fits_name=fits_name,
    #             result=None,
    #             selector=selector,
    #         ).keys()
    #     ),
    # )

    # If user escapes selection, write null row and exit
    if ap_box is None:
        print("[PHOT] User escaped selection; writing null row.")
        row = _build_csv_row(
            target_name=target,
            obs_id=observation_id,
            filt=filt,
            fits_name=fits_name,
            result=None,
            selector=selector,
        )
        append_row(filepath=phot_csv, row=row)
        return

    # Save PNG
    try:
        png_path = _build_screenshot_path(config=config, fits_path=fits_path, target=target)
        ui.fig.savefig(png_path, dpi=150, bbox_inches="tight")
        print(f"[PHOT] PNG exported: {png_path}")
    except Exception as e:
        print(f"[PHOT] WARN: PNG export failed: {e}")

    # Run asteroid photometry
    try:
        res: PhotometryResult = pt.perform_trail_photometry(ap_box, ann_box, debug=True)
    except Exception as e:
        print(f"[PHOT] WARN: photometry failed: {e}")
        print("[PHOT] Writing null row to avoid retrying this frame.")

        row = _build_csv_row(
            target_name=target,
            obs_id=observation_id,
            filt=filt,
            fits_name=fits_name,
            result=None,
            selector=selector,
        )

        append_row(
            filepath=phot_csv,
            row=row,
        )
        return

    # Compute apcorr from selected stars (if any) BEFORE writing main CSV row
    apcorr_mag = 0.0
    apcorr_mag_err = 0.0
    f_apcorr = 1.0

    apcorr_dir = Path(config.get("PHOTOMETRY", "APCORR_DIRECTORY", fallback="")).expanduser()
    if str(apcorr_dir).strip() == "":
        # safe fallback
        apcorr_dir = Path(phot_csv).expanduser().parent / "apcorr"

    if srclist_path is not None:
        _, apcorr_csv_path, apcorr_tuple = _write_apcorr_star_csv(
            ui=ui,
            pt=pt,
            srclist_path=srclist_path,
            apcorr_dir=apcorr_dir,
            fits_name=fits_name,
            observation_id=observation_id,
            filt=filt,
            target=target,
        )
        if apcorr_tuple is not None:
            apcorr_mag, apcorr_mag_err = apcorr_tuple
            # multiplicative factor corresponding to apcorr_mag
            try:
                f_apcorr = float(10.0 ** (-float(apcorr_mag) / 2.5))
            except Exception:
                f_apcorr = 1.0
    else:
        print("[PHOT] NOTE: no SRCLIST available; apcorr not computed.")

    # Build row, then augment with apcorr
    row = _build_csv_row(
        target_name=target,
        obs_id=observation_id,
        filt=filt,
        fits_name=fits_name,
        result=res,
        selector=selector,
    )
    # Propagate screening/catalogue context into photometry_output.csv
    row["v_mag_1"] = extract_row_value(screening_row, V_MAG_1_COL)
    row["v_mag_1_corrected"] = extract_row_value(screening_row, V_MAG_1_CORRECTED_COL)
    row["mlim_obs"] = extract_row_value(screening_row, MLIM_OBS_COL)

    # -----------------------------------------------------------------
    # Apply rate-domain corrections consistently (CoI, TDS, C2, apcorr)
    # -----------------------------------------------------------------
    # 0) Inputs from photometry result
    # R_net_h = float(res.count_rate) if res.count_rate is not None else float("nan")
    # R_net_h_err = float(res.count_rate_err) if res.count_rate_err is not None else float("nan")
    # bkg_per_pix = float(res.bkg_per_pix) if res.bkg_per_pix is not None else 0.0
    # A_h_eff = float(res.A_ap_eff) if res.A_ap_eff is not None else float("nan")

    R_net_h = float(res.count_rate) if res.count_rate is not None else float("nan")
    R_net_h_err = float(res.count_rate_err) if res.count_rate_err is not None else float("nan")

    # perform_trail_photometry returns background in image units.
    # For COUNTS images this is counts/pix, so convert to counts/s/pix
    # before building a total rate for CoI.
    bkg_counts_per_pix = float(res.bkg_per_pix) if res.bkg_per_pix is not None else 0.0

    exptime = float(getattr(res, "exptime", np.nan))
    if not np.isfinite(exptime) or exptime <= 0:
        exptime = float(getattr(hduw, "texp", np.nan))

    if not np.isfinite(exptime) or exptime <= 0:
        exptime = float(hduw.hdu.header.get("EXPOSURE", np.nan))

    bkg_rate_per_pix = (
        bkg_counts_per_pix / exptime
        if np.isfinite(bkg_counts_per_pix) and np.isfinite(exptime) and exptime > 0
        else 0.0
    )
    

    A_h_eff = float(res.A_ap_eff) if res.A_ap_eff is not None else float("nan")

    # # 1) Standard 6"-equivalent box height (full width = 12")
    # #    We approximate A_6_eff by scaling A_h_eff by the ratio of heights,
    # #    keeping the same trail length/width.
    # try:
    #     from src.photometry.phot import arcsec_to_pix  # type: ignore
    #     height6_pix = float(arcsec_to_pix(hduw, 12.0))
    # except Exception:
    #     # fallback: assume binned ~0.95"/pix; 12" ~ 12.6 pix
    #     height6_pix = 12.6

    trail_height_pix_eff = float(getattr(ap_box, "h", np.nan))
    if np.isfinite(A_h_eff) and np.isfinite(trail_height_pix_eff) and trail_height_pix_eff > 0:
        A6_eff = float(A_h_eff * (height6_pix / trail_height_pix_eff))
    else:
        A6_eff = float("nan")

    # # 2) C1: not yet implemented interactively; default to 1.0
    # C1 = float(config.get("PHOTOMETRY", "C1", fallback="1.0") or 1.0)
    # R_net_6 = float(R_net_h * C1) if np.isfinite(R_net_h) else float("nan")
    # R_net_6_err = float(R_net_h_err * C1) if np.isfinite(R_net_h_err) else float("nan")

    # 2) C1: only needed if the selected trail-box height is smaller than
    #    the standard 6 arcsec radius equivalent box height.
    trail_width_pix_eff = float(getattr(ap_box, "w", np.nan))
    # trail_theta_rad_eff = float(getattr(ap_box, "theta", 0.0))
    trail_theta_rad_eff = angle_to_rad(getattr(ap_box, "theta", 0.0))

    trail_semi_out_pix_eff = (
        float((ann_box.w_out - ann_box.w_in) / 2.0)
        if ann_box is not None
        else float("nan")
    )

    C1, C1_err, c1_csv_path = _compute_c1_if_needed(
        config=config,
        hduw=hduw,
        pt=pt,
        srclist_path=srclist_path,
        fits_name=fits_name,
        observation_id=observation_id,
        filt=filt,
        target=target,
        trail_width_pix=trail_width_pix_eff,
        trail_height_pix=trail_height_pix_eff,
        trail_theta_rad=trail_theta_rad_eff,
        trail_semi_out_pix=trail_semi_out_pix_eff,
        height6_pix=height6_pix,
    )

    R_net_6 = float(R_net_h * C1) if np.isfinite(R_net_h) else float("nan")
    # Propagate Rh -> R6 = C1 * Rh.
    # Rh (asteroid photometry) and C1 (calibration stars) are treated
    # as independent measurements:
    #
    # Var(R6) = (C1 * sigma_Rh)^2 + (Rh * sigma_C1)^2
    R_net_6_err = float(np.sqrt((C1 * R_net_h_err)**2 + (R_net_h * C1_err)**2)) if all(np.isfinite(v) for v in (R_net_h, R_net_h_err, C1, C1_err)) else float("nan")

    print(
        f"[PHOT][C1][ERR] "
        f"Rh={R_net_h:.6g} +/- {R_net_h_err:.6g} "
        f"C1={C1:.6g} +/- {C1_err:.6g} "
        f"-> R6={R_net_6:.6g} +/- {R_net_6_err:.6g}"
    )
    # 3) Build total (source+background) rate in the 6"-equivalent aperture
    # R_tot_6 = float("nan")
    # if np.isfinite(R_net_6) and np.isfinite(A6_eff):
    #     R_tot_6 = float(R_net_6 + bkg_per_pix * A6_eff)
    # 3) Build total source+background RATE in the 6"-equivalent aperture
    R_bkg_6 = float("nan")
    R_tot_6 = float("nan")

    if np.isfinite(bkg_rate_per_pix) and np.isfinite(A6_eff):
        R_bkg_6 = float(bkg_rate_per_pix * A6_eff)

    if np.isfinite(R_net_6) and np.isfinite(R_bkg_6):
        R_tot_6 = float(R_net_6 + R_bkg_6)

    if (
        np.isfinite(R_net_h)
        and R_net_h != 0
        and np.isfinite(C1)
        and C1 != 0
    ):
        print(
            f"[PHOT][C1][ERR] fractional: "
            f"Rh={R_net_h_err / abs(R_net_h):.4f}, "
            f"C1={C1_err / abs(C1):.4f}, "
            f"R6={R_net_6_err / abs(R_net_6):.4f}"
        )

    # 4) CoI correction on total rate
    hdr = hduw.hdu.header
    frame_time = float(hdr["FRAMTIME"]) * 1e-3  # ms -> s
    dead_fraction = float(hdr.get("DEADFRAC", 0.0))
    
    coi_coeffs = _parse_float_list(config.get("PHOTOMETRY", "COI_F_COEFFS", fallback=""))
    R_tot_6_coi = float("nan")
    coi_factor = float("nan")
    R_net_6_coi = float("nan")
    R_net_6_coi_err = float("nan")
    R_bkg_6_coi = float("nan")

    sigma_bkg_rate_per_pix = float(res.bkg_rms_per_pix / (exptime * np.sqrt(res.A_bg_eff)))
    R_bkg_6_err = float(A6_eff * sigma_bkg_rate_per_pix)
    cov_R6_bkg6 = float(-C1 * A_h_eff * A6_eff * sigma_bkg_rate_per_pix**2)

    if np.isfinite(R_tot_6) and np.isfinite(frame_time):
        try:
            R_tot_6 = R_net_6 + R_bkg_6
            R_tot_6_coi = apply_coi_correction(R_tot_6, frame_time, dead_fraction)
            R_bkg_6_coi = apply_coi_correction(R_bkg_6, frame_time, dead_fraction)
            R_net_6_coi = float(R_tot_6_coi - R_bkg_6_coi)

            coi_factor = (
                float(R_net_6_coi / R_net_6)
                if np.isfinite(R_net_6) and R_net_6 > 0
                else float("nan")
            )
            # propagate statistical uncertainty via derivative dRcorr/dRraw
            d_dR = coi_derivative_dR(
                R_raw=R_tot_6,
                frame_time=float(frame_time),
                dead_fraction=float(dead_fraction),
                f_coeffs=coi_coeffs,
            )
            d_tot = coi_derivative_dR(R_tot_6, frame_time, dead_fraction)
            d_bkg = coi_derivative_dR(R_bkg_6, frame_time, dead_fraction)  
            if np.isfinite(d_dR) and np.isfinite(R_net_6_err):
                # treat sigma(R_tot_6) ~ sigma(R_net_6) (to be reviewed by Simon if we go for this simplification)
                # R_net_6_coi_err = float(abs(d_dR) * R_net_6_err)
                var_R_net_6_coi = d_tot**2 * R_net_6_err**2 + (d_tot - d_bkg)**2 * R_bkg_6_err**2 + 2.0 * d_tot * (d_tot - d_bkg) * cov_R6_bkg6
                R_net_6_coi_err = float(np.sqrt(max(var_R_net_6_coi, 0.0)))
        except Exception as e:
            print(f"[PHOT] WARN: CoI correction failed; using raw rates (reason: {e})")
            R_tot_6_coi = R_tot_6
            coi_factor = 1.0
            R_net_6_coi = R_net_6
            R_net_6_coi_err = R_net_6_err
    else:
        # no CoI possible
        R_tot_6_coi = R_tot_6
        coi_factor = 1.0
        R_net_6_coi = R_net_6
        R_net_6_coi_err = R_net_6_err

    # 5) TDS correction retrieved from source list

    # 6) C2 correction (config.ini); for non-UV filters set C2=1
    C2 = float(config.get("PHOTOMETRY", "C2", fallback="1.0") or 1.0)
    C2_err = float(config.get("PHOTOMETRY", "C2_ERR", fallback="0.0") or 0.0)

    # ================================================================
    # 7) FINAL SCIENCE RATE
    #
    #     R_f = R_6,CoI * C2 * C_TDS
    #
    # This is the rate used for the scientific AB magnitude.
    # APCORR IS NOT USED IN THIS BRANCH.
    # ================================================================
    R_final = (
        float(R_net_6_coi * tds_corr * C2)
        if np.isfinite(R_net_6_coi)
        else float("nan")
    )

    # TDS is treated as fixed (no TDS uncertainty).
    # C2 uncertainty is the empirical calibration scatter (MAD).
    R_final_err = (
        float(np.sqrt(
            (C2 * tds_corr * R_net_6_coi_err) ** 2
            + (R_net_6_coi * tds_corr * C2_err) ** 2
        ))
        if all(np.isfinite(v) for v in (
            R_net_6_coi,
            R_net_6_coi_err,
            tds_corr,
            C2,
            C2_err,
        ))
        else float("nan")
    )


    # ================================================================
    # DIAGNOSTIC APCORR BRANCH
    #
    # Stored only for traceability / possible future diagnostics.
    # These quantities MUST NOT be used to derive:
    #   - count_rate_final
    #   - count_rate_final_err
    #   - mag_ab
    #   - mag_err
    # ================================================================

    # Empirical uncertainty of apcorr in magnitude space
    try:
        apcorr_mag_err_val = float(apcorr_mag_err)
    except (TypeError, ValueError):
        apcorr_mag_err_val = 0.0

    if not np.isfinite(apcorr_mag_err_val):
        apcorr_mag_err_val = 0.0

    # f_apcorr = 10^(-apcorr_mag / 2.5)
    # Propagate sigma(apcorr_mag) -> sigma(f_apcorr)
    f_apcorr_err = (
        float(
            abs(f_apcorr)
            * (np.log(10.0) / 2.5)
            * apcorr_mag_err_val
        )
        if np.isfinite(f_apcorr)
        else float("nan")
    )

    R_final_apcorr = (
        float(R_final * f_apcorr)
        if np.isfinite(R_final) and np.isfinite(f_apcorr)
        else float("nan")
    )

    R_final_apcorr_err = (
        float(np.sqrt(
            (f_apcorr * R_final_err) ** 2
            + (R_final * f_apcorr_err) ** 2
        ))
        if all(np.isfinite(v) for v in (
            R_final,
            R_final_err,
            f_apcorr,
            f_apcorr_err,
        ))
        else float("nan")
    )


    # ================================================================
    # 8) CONVERT TO AB MAGNITUDES
    # ================================================================
    zp_ab = (
        res.zp_ab
        if res.zp_ab is not None
        else (
            pt.zero_point
            if pt.zero_point is not None
            else None
        )
    )

    # ----------------
    # Scientific branch
    # ----------------
    mag_ab = None
    mag_ab_err = None

    if (
        zp_ab is not None
        and np.isfinite(R_final)
        and R_final > 0
    ):
        mag_ab = float(
            -2.5 * np.log10(R_final) + float(zp_ab)
        )

        if (
            np.isfinite(R_final_err)
            and R_final_err >= 0
        ):
            mag_ab_err = float(
                AB_PROP_COEFF
                * R_final_err
                / R_final
            )


    # ----------------
    # Diagnostic APCORR branch only
    # ----------------
    mag_ab_apcorr = None
    mag_ab_apcorr_err = None

    if (
        zp_ab is not None
        and np.isfinite(R_final_apcorr)
        and R_final_apcorr > 0
    ):
        mag_ab_apcorr = float(
            -2.5 * np.log10(R_final_apcorr)
            + float(zp_ab)
        )

        if (
            np.isfinite(R_final_apcorr_err)
            and R_final_apcorr_err >= 0
        ):
            mag_ab_apcorr_err = float(
                AB_PROP_COEFF
                * R_final_apcorr_err
                / R_final_apcorr
            )


    # ================================================================
    # 9) STORE OUTPUTS
    # ================================================================

    # Zero point
    row["zp_ab"] = (
        float(zp_ab)
        if zp_ab is not None
        else row.get("zp_ab")
    )

    # ----------------
    # SCIENCE OUTPUTS
    # ----------------
    row["mag_ab"] = mag_ab
    row["mag_err"] = mag_ab_err

    row["tds_corr"] = tds_corr

    row["c2"] = C2
    row["c2_err"] = C2_err

    row["c1"] = C1
    row["c1_err"] = C1_err
    row["c1_detail_csv"] = (
        str(c1_csv_path)
        if c1_csv_path is not None
        else None
    )

    row["coi_factor"] = coi_factor

    # Rate-domain correction chain
    row["rate_tot_6"] = R_tot_6
    row["rate_tot_6_coi"] = R_tot_6_coi

    row["rate_bkg_6"] = R_bkg_6
    row["rate_bkg_6_coi"] = R_bkg_6_coi

    # Critical intermediate product:
    # allows R_f and m_AB to be recomputed later without rerunning photometry
    row["count_rate_coi"] = R_net_6_coi
    row["count_rate_coi_err"] = R_net_6_coi_err

    # Official final science rate:
    # R_f = R_6,CoI * C2 * C_TDS
    row["count_rate_final"] = R_final
    row["count_rate_final_err"] = R_final_err


    # ----------------
    # DIAGNOSTIC APCORR OUTPUTS ONLY
    # ----------------
    row["f_apcorr"] = f_apcorr
    row["f_apcorr_err"] = f_apcorr_err

    row["apcorr_mag"] = apcorr_mag
    row["apcorr_mag_err"] = apcorr_mag_err_val

    row["count_rate_final_apcorr"] = R_final_apcorr
    row["count_rate_final_apcorr_err"] = R_final_apcorr_err

    row["mag_ab_apcorr"] = mag_ab_apcorr
    row["mag_ab_apcorr_err"] = mag_ab_apcorr_err


    # ================================================================
    # 10) STORE FINAL APERTURE GEOMETRY
    # ================================================================

    # Derive actual geometry from the final apertures
    trail_height_pix = float(
        getattr(ap_box, "h", np.nan)
    )
    trail_width_pix = float(
        getattr(ap_box, "w", np.nan)
    )

    # Annulus thickness in pixels (semi_out),
    # and any inner gap (semi_in)
    semi_out = (
        float((ann_box.w_out - ann_box.w_in) / 2.0)
        if ann_box is not None
        else np.nan
    )

    semi_in = (
        float((ann_box.w_in - ap_box.w) / 2.0)
        if ann_box is not None
        else 0.0
    )
    semi_in = max(0.0, semi_in)

    # Store these (override selector-based fields)
    row["trail_height_pix"] = trail_height_pix
    row["trail_semi_out_pix"] = semi_out
    row["trail_semi_in_pix"] = semi_in


    # ================================================================
    # 11) WRITE PHOTOMETRY ROW
    # ================================================================
    append_row(
        filepath=phot_csv,
        row=row,
    )