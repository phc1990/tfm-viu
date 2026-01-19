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
from src.photometry.phot import PhotTable, PhotometryResult
from src.photometry.ui import UI, TrailSelector
from src.screening.xsa import convert_filter_name_to_xsa_name

from common import (
    OBS_ID_COLS,
    TARGET_COLS,
    FILTER_COLS,
    FITS_FILE_COLS,
    POS1_DEC_COLS,
    POS1_RA_COLS,
    POS2_DEC_COLS,
    POS2_RA_COLS,
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

# def ensure_srclist_downloaded(config, observation_id: str, filt: str, dest_dir: Path) -> None:
#     crawler = HttpCurlCrawler(
#         download_directory=str(dest_dir.parent.parent),   # raíz DOWNLOAD_DIRECTORY
#         base_url=config["INPUT"]["BASE_URL"],
#         regex_pattern=config["PHOTOMETRY"].get("SRCLIST_REGEX", r"^.*?SWSRLI.*?\.FTZ$"),
#     )
#     crawler.crawl(observation_id=observation_id, filters=[filt])

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
    band = PhotTable._filter_to_band(filt)
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
def _mad_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return 0.0
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(1.4826 * mad)


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
        "target_name": target_name,
        "observation_id": obs_id,
        "filter": filt,
        "fits_name": fits_name,

        "mag_ab": None,
        "mag_err": None,
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

        "trail_height_pix": getattr(selector, "height", None) if selector else None,
        "trail_semi_out_pix": getattr(selector, "semi_out", None) if selector else None,
        "trail_semi_in_pix": getattr(selector, "semi_in", None) if selector else None,

        # NEW (always present)
        "apcorr_mag": None,
        "mag_ab_apcorr": None,
        "mag_ab_apcorr_err": None,
    }

    if result is None:
        return row
    
    bkg_counts_ap = None
    bkg_counts_ap_err = None

    if (result.bkg_per_pix is not None) and (result.A_ap_eff is not None):
        bkg_counts_ap = float(result.bkg_per_pix) * float(result.A_ap_eff)

    if (result.bkg_rms_per_pix is not None) and (result.A_ap_eff is not None):
        bkg_counts_ap_err = float(result.bkg_rms_per_pix) * float(np.sqrt(float(result.A_ap_eff)))

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

    # pos1/pos2 markers (non-blocking)

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
                # base = "https://nxsa.esac.esa.int/nxsa-sl/servlet/data-action-aio"
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


    selector = TrailSelector(height=5.0, semi_out=5.0, finalize_on_click=False)

    try:
        sel = ui.select_trail(selector)
    except Exception as e:
        print(f"[PHOT] WARN: UI selection failed: {e}")
        return

    ap_box, ann_box = _normalize_selection_to_ap_ann(sel)

    # Main photometry CSV: migrate header upfront (append_row rejects unknown keys)
    phot_csv = config["PHOTOMETRY"]["FILEPATH"]
    ensure_csv_has_fields(
        phot_csv,
        list(
            _build_csv_row(
                target_name=target,
                obs_id=observation_id,
                filt=filt,
                fits_name=fits_name,
                result=None,
                selector=selector,
            ).keys()
        ),
    )

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
        return

    # Compute apcorr from selected stars (if any) BEFORE writing main CSV row
    apcorr_mag = 0.0
    apcorr_mag_err = 0.0

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

    # derive actual geometry from the final apertures
    trail_height_pix = float(getattr(ap_box, "h", np.nan))
    trail_width_pix = float(getattr(ap_box, "w", np.nan))

    # annulus thickness in pixels (semi_out), and any inner gap (semi_in)
    semi_out = float((ann_box.w_out - ann_box.w_in) / 2.0) if ann_box is not None else np.nan
    semi_in = float((ann_box.w_in - ap_box.w) / 2.0) if ann_box is not None else 0.0
    semi_in = max(0.0, semi_in)

    # store these (override selector-based fields)
    row["trail_height_pix"] = trail_height_pix
    row["trail_semi_out_pix"] = semi_out
    row["trail_semi_in_pix"] = semi_in

    # store aperture corrected mag and error
    row["apcorr_mag"] = apcorr_mag
    if res.mag_ab is not None:
        row["mag_ab_apcorr"] = float(res.mag_ab) + float(apcorr_mag)
        base_err = float(res.mag_err) if res.mag_err is not None else 0.0
        row["mag_ab_apcorr_err"] = float(np.hypot(base_err, apcorr_mag_err))
    else:
        row["mag_ab_apcorr"] = None
        row["mag_ab_apcorr_err"] = None

    append_row(filepath=phot_csv, row=row)
