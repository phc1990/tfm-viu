#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
from configparser import ConfigParser

from src.photometry.hdu import HDUW
from src.photometry.ui import UI, TrailSelector
from src.photometry.phot import PhotTable, arcsec_to_pix, compute_c2_for_star


# -----------------------------
# small utils
# -----------------------------
def _parse_list_csv(s: str) -> List[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def _glob_simag_ftz(directory: Path) -> List[Path]:
    # Only SIMAG* are 2D images
    # Example: P0781040101OMS008FSIMAGL000.FTZ, P...OMX...LSIMAGL000.FTZ
    return sorted(set(directory.glob("*SIMAG*.FTZ")) | set(directory.glob("*SIMAG*.ftz")))


def _glob_swsrli_ftz(directory: Path) -> List[Path]:
    # Only SWSRLI* are source lists (tables) for overlay
    # Example: P0781040101OMS008FSWSRLIL000.FTZ
    return sorted(set(directory.glob("*SWSRLI*.FTZ")) | set(directory.glob("*SWSRLI*.ftz")))



def _extract_obsid(name: str) -> Optional[str]:
    m = re.search(r"(\d{10})", name)
    return m.group(1) if m else None


def _infer_filter_from_header(hduw: HDUW) -> str:
    hdr = hduw.hdu.header
    for key in ("FILTER", "FILTNAM1", "OMFILTER", "FILTERID"):
        v = hdr.get(key)
        if v is not None:
            s = str(v).strip().upper()
            if s:
                return s
    return "UNKNOWN"


def _robust_stats(values: List[float]) -> Dict[str, float]:
    v = np.array(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return dict(n=0, median=np.nan, mean=np.nan, std=np.nan, mad=np.nan)
    med = float(np.median(v))
    mean = float(np.mean(v))
    std = float(np.std(v, ddof=1)) if v.size >= 2 else np.nan
    mad = float(1.4826 * np.median(np.abs(v - med))) if v.size >= 2 else np.nan
    return dict(n=int(v.size), median=med, mean=mean, std=std, mad=mad)


# def _extract_exposure_token(name: str) -> Optional[str]:
#     """
#     Examples:
#       P0781040101OMS008FSIMAGL000.FTZ -> P0781040101OMS008F
#       P0781040101OMS010FSIMAGL000.FTZ -> P0781040101OMS010F
#       P0781040101OMX000LSIMAGL000.FTZ -> P0781040101OMX000L
#     """
#     up = name.upper()
#     m = re.search(r"(P\d{10}OMS\d{3}F)", up)
#     if m:
#         return m.group(1)
#     m = re.search(r"(P\d{10}OMX\d{3}[A-Z])", up)
#     if m:
#         return m.group(1)
#     return None

def _extract_exposure_token(name: str) -> Optional[str]:
    """
    Examples:
      P0781040101OMS008FSIMAGL000.FTZ -> P0781040101OMS008
      P0700182001OMS008SIMAGE0000.FTZ -> P0700182001OMS008
      P0781040101OMX000LSIMAGL000.FTZ -> P0781040101OMX000
    """
    up = name.upper()

    m = re.search(r"(P\d{10}OMS\d{3})", up)
    if m:
        return m.group(1)

    m = re.search(r"(P\d{10}OMX\d{3})", up)
    if m:
        return m.group(1)

    return None

def _extract_band_from_name(path: Path) -> Optional[str]:
    up = path.name.upper()
    m = re.search(r"SIMAG([LMSUVB])", up)
    if m:
        return PhotTable._filter_to_band(m.group(1))
    m = re.search(r"SWSRLI([LMSUVB])", up)
    if m:
        return PhotTable._filter_to_band(m.group(1))
    return None

def _resolve_band(path: Path, hduw: Optional[HDUW] = None) -> Optional[str]:
    band = _extract_band_from_name(path)
    if band is not None:
        return band

    if hduw is not None:
        filt = _infer_filter_from_header(hduw)
        if filt and filt != "UNKNOWN":
            return PhotTable._filter_to_band(filt)

    return None

def _find_swsrli_for_simag(simag_path: Path, swsrli_files: List[Path], band: Optional[str] = None) -> Optional[Path]:
    token = _extract_exposure_token(simag_path.name)
    band = band or _extract_band_from_name(simag_path)
    if token is None or band is None:
        return None

    for p in swsrli_files:
        up = p.name.upper()
        if token in up and "SWSRLI" in up and _extract_band_from_name(p) == band:
            return p

    return None
# -----------------------------
# main action
# -----------------------------
def action_phot_c2(config: ConfigParser) -> None:
    if "C2" not in config:
        raise KeyError("Missing [C2] section in config.ini")

    c2 = config["C2"]

    ftz_dir = Path(c2.get("FTZ_DIRECTORY", "")).expanduser()
    out_file = Path(c2.get("OUTPUT_FILE", "")).expanduser()
    out_detail = Path(c2.get("OUTPUT_DETAIL_FILE", str(out_file.with_suffix(".detail.csv")))).expanduser()

    filters = [x.strip().upper() for x in _parse_list_csv(c2.get("FILTERS", ""))]
    n_stars = int(c2.get("N_STARS", "5"))

    r_small_arcsec = float(c2.get("R_SMALL_ARCSEC", "6"))
    r_large_arcsec = float(c2.get("R_LARGE_ARCSEC", "35"))

    box_len_arcsec = float(c2.get("BOX_LENGTH_ARCSEC", "70"))

    ann_small_arcsec = float(c2.get("ANNULUS_SEMIOUT_SMALL_ARCSEC", "6"))
    ann_large_arcsec = float(c2.get("ANNULUS_SEMIOUT_LARGE_ARCSEC", "10"))

    srclist_dir_raw = c2.get("SRCLIST_DIRECTORY", "").strip()
    srclist_dir = Path(srclist_dir_raw).expanduser() if srclist_dir_raw else None

    if not ftz_dir.exists():
        raise FileNotFoundError(f"[C2] FTZ_DIRECTORY does not exist: {ftz_dir}")
    
    simag_files = _glob_simag_ftz(ftz_dir)
    swsrli_files = _glob_swsrli_ftz(ftz_dir)

    if not simag_files:
        raise FileNotFoundError(f"[C2] No SIMAG FTZ images found in {ftz_dir}")


    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_detail.parent.mkdir(parents=True, exist_ok=True)

    print(f"[C2] OUTPUT_FILE        = {out_file}")
    print(f"[C2] OUTPUT_DETAIL_FILE = {out_detail}")


    # collect across all files
    c2_by_filter: Dict[str, List[float]] = {}
    detail_rows: List[Dict[str, Any]] = []

    for ftz in simag_files:
        filt="UNKNOWN"
        # --- load image ---
        print(f"File = {ftz}")
        hduw = HDUW(file=str(ftz))

        # Sanity-check image shape without casting (avoid dtype warnings)
        raw = getattr(hduw, "data", None)
        if raw is None:
            raw = getattr(getattr(hduw, "hdu", None), "data", None)
        if raw is None or getattr(raw, "ndim", 0) != 2:
            raise ValueError("UI: expected a 2-D image in the HDU.")

        phot = PhotTable(hduw)

        filters_raw = [x.strip().upper() for x in _parse_list_csv(c2.get("FILTERS", ""))]
        filters_band = [PhotTable._filter_to_band(f) for f in filters_raw]

        band = _resolve_band(ftz, hduw) or "UNKNOWN"
        if filters_band and band not in filters_band:
            print(f"[C2] Skipping {ftz.name} band={band} (not in FILTERS={filters_band})")
            continue

        # per-file arcsec->pix (handles binning/platescale)
        height6_pix = float(arcsec_to_pix(hduw, 2.0 * r_small_arcsec))
        height35_pix = float(arcsec_to_pix(hduw, 2.0 * r_large_arcsec))
        semi6_pix = float(arcsec_to_pix(hduw, ann_small_arcsec))
        semi35_pix = float(arcsec_to_pix(hduw, ann_large_arcsec))
        width_pix0 = float(arcsec_to_pix(hduw, box_len_arcsec))

        selector = TrailSelector(height=height6_pix, semi_out=semi6_pix, finalize_on_click=False)
        selector.width = width_pix0

        print(f"[C2] geometry: width={width_pix0:.2f}px ({box_len_arcsec:.1f}\") height6={height6_pix:.2f}px height35={height35_pix:.2f}px")
        
        # ui = UI(hduw)
        # ui.c2_height35_pix = height35_pix
        # ui.c2_semi35_pix = semi35_pix
        ui = UI(hduw)
        ui.c2_height35_pix = height35_pix
        ui.c2_semi35_pix = semi35_pix
        ui.calib_auto_slot = True
        ui.calib_max_slots = n_stars
        ui.keep_calib_boxes = True

        # SRCLIST overlay (optional but recommended)
        swsrli = _find_swsrli_for_simag(ftz, swsrli_files, band=band)
        if swsrli is not None and swsrli.exists():
            try:
                ui.add_srclist_overlay(swsrli)
                print(f"[C2] SWSRLI overlay: {swsrli.name}")
            except Exception as e:
                print(f"[C2] WARN: could not overlay SWSRLI ({swsrli}): {e}")
                
        else:
            print(f"[C2] WARN: no SWSRLI found for {ftz.name} (no yellow sources).")

        title = (
            f"C2 calibration — {ftz.name} band={band}\n"
            f"1) Dibuja dirección/longitud (define theta+width). 2) Pulsa 'A' para seleccionar cada estrella (máx. {n_stars}).\n"            f"Height6={height6_pix:.1f}px  Height35={height35_pix:.1f}px"
        )
        try:
            ui.fig.suptitle(title, fontsize=10)
        except Exception:
            # fallback
            ui.ax.set_title(title, fontsize=10)
        ui.update()

        # Esto bloquea hasta que pulses Enter (y ahí se cierra la figura)
        ui.select_trail(selector)

        #Guardemos screenshot para tesis
        png_name = f"{Path(ftz).name}_c2_factor.png"
        png_path = out_detail.parent / png_name

        try:
            ui.fig.savefig(png_path, dpi=200, bbox_inches="tight")
            print(f"[C2] Snapshot saved: {png_path}")
        except Exception as e:
            print(f"[C2] WARN: could not save snapshot {png_path}: {e}")

        # Al volver aquí, las estrellas quedan en:
        sels = ui.calib_star_selections

        sels: Dict[int, Dict[str, Any]] = getattr(ui, "calib_star_selections", {}) or {}
        if not sels:
            print(f"[C2] No calibration stars saved in {ftz.name}; skipping file.")
            continue

        for slot, d in sorted(sels.items()):
            try:
                x = float(d["x"])
                y = float(d["y"])
                # width_pix = float(d.get("width", width_pix0))
                width_pix = width_pix0
                theta = float(d.get("theta", selector.theta if getattr(selector, "theta", None) is not None else 0.0))

                meas = compute_c2_for_star(
                    phot=phot,
                    x=x, y=y,
                    width_pix=width_pix,
                    theta_rad=theta,
                    height6_pix=height6_pix,
                    height35_pix=height35_pix,
                    semi_out6_pix=semi6_pix,
                    semi_out35_pix=semi35_pix,
                    bg_center=None,
                    debug=False,
                )
                meas.slot = int(slot)

                snr6 = float(meas.rate6 / meas.err6) if np.isfinite(meas.rate6) and np.isfinite(meas.err6) and meas.err6 > 0 else float("nan")
                snr35 = float(meas.rate35 / meas.err35) if np.isfinite(meas.rate35) and np.isfinite(meas.err35) and meas.err35 > 0 else float("nan")

                c2_by_filter.setdefault(band, []).append(meas.c2)

                detail_rows.append(dict(
                    ftz=str(ftz),
                    band=band,
                    slot=meas.slot,
                    x=x, y=y,
                    width_pix=width_pix,
                    theta_rad=theta,

                    rate6=meas.rate6, err6=meas.err6, snr6=snr6,
                    rate35=meas.rate35, err35=meas.err35, snr35=snr35,
                    c2=meas.c2, #c2_err=meas.c2_err,

                    aper_sum_6=meas.aper_sum_6,
                    bkg_per_pix_6=meas.bkg_per_pix_6,
                    bkg_rms_per_pix_6=meas.bkg_rms_per_pix_6,
                    A_ap_eff_6=meas.A_ap_eff_6,
                    A_bg_eff_6=meas.A_bg_eff_6,
                    unit_kind_6=meas.unit_kind_6,
                    exptime_6=meas.exptime_6,

                    aper_sum_35=meas.aper_sum_35,
                    bkg_per_pix_35=meas.bkg_per_pix_35,
                    bkg_rms_per_pix_35=meas.bkg_rms_per_pix_35,
                    A_ap_eff_35=meas.A_ap_eff_35,
                    A_bg_eff_35=meas.A_bg_eff_35,
                    unit_kind_35=meas.unit_kind_35,
                    exptime_35=meas.exptime_35,
                ))

                print(f"[C2] {ftz.name} A{meas.slot}: C2={meas.c2:.6f} | R6={meas.rate6:.6g} +/- {meas.err6:.3g} (S/N={snr6:.1f}) | R35={meas.rate35:.6g} +/- {meas.err35:.3g} (S/N={snr35:.1f})")

            except Exception as e:
                print(f"[C2] WARN: slot A{slot} failed in {ftz.name}: {e}")

        # --- write detail ---
        if detail_rows:
            fieldnames = sorted({k for r in detail_rows for k in r.keys()})

            write_header = (not out_detail.exists()) or (out_detail.stat().st_size == 0)
            with out_detail.open("a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                if write_header:
                    w.writeheader()
                w.writerows(detail_rows)

            print(f"[C2] Appended detail: {out_detail}")
        else:
            print("[C2] No detail rows produced.")

    # --- write summary per band ---
    summary_rows: List[Dict[str, Any]] = []
    for band_key, vals in sorted(c2_by_filter.items()):
        st = _robust_stats(vals)
        summary_rows.append(dict(
            band=band_key,
            n=st["n"],
            c2_median=st["median"],
            c2_mean=st["mean"],
            c2_std=st["std"],
            c2_mad=st["mad"],
            r_small_arcsec=r_small_arcsec,
            r_large_arcsec=r_large_arcsec,
            ann_small_arcsec=ann_small_arcsec,
            ann_large_arcsec=ann_large_arcsec,
        ))

    if summary_rows:
        # Fixed schema (do NOT change)
        fieldnames = [
            "band",
            "n",
            "c2_median",
            "c2_mean",
            "c2_std",
            "c2_mad",
            "r_small_arcsec",
            "r_large_arcsec",
            "ann_small_arcsec",
            "ann_large_arcsec",
        ]

        write_header = (not out_file.exists()) or (out_file.stat().st_size == 0)
        with out_file.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if write_header:
                w.writeheader()
            w.writerows(summary_rows)

        print(f"[C2] Appended summary: {out_file}")
    else:
        print("[C2] No summary rows produced (no C2 computed).")
