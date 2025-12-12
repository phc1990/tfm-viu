#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
photometry.py — mosaic-first trail photometry with a single CSV writer
"""

from __future__ import annotations

import argparse
from configparser import ConfigParser
import csv
import math
from pathlib import Path
from typing import Optional, Dict, Tuple

import numpy as np

import warnings
warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in subtract",
    category=RuntimeWarning,
    module=r"matplotlib\.colors"
)


from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs.utils import skycoord_to_pixel

from src.photometry.hdu import HDUW
from src.photometry.phot import PhotTable, PhotometryResult
from src.photometry.ui import UI, TrailSelector

from download import (
    ensure_om_srclist_or_obsmlist,    # (obs_id) -> (Path, "OBSMLI")
    download_om_mosaic_sky_image,     # (obs_id, filter_code) -> Path
)


def _plot_pos1_pos2(ui, w, pos1, pos2):
    """Convierte RA/Dec a píxeles y pinta SIEMPRE pos1 y pos2 en la imagen."""
    xs, ys, labels = [], [], []
    if w is None:
        print("[PHOT][UI] WARN: sin WCS; no puedo convertir RA/Dec → píxel.")
        return

    ra_list  = [float(pos1[0]), float(pos2[0])]
    dec_list = [float(pos1[1]), float(pos2[1])]
    sc = SkyCoord(ra_list * u.deg, dec_list * u.deg, frame="icrs")
    X, Y = w.world_to_pixel(sc)

    for (xi, yi, name) in zip(np.atleast_1d(X), np.atleast_1d(Y), ("pos1", "pos2")):
        if np.isfinite(xi) and np.isfinite(yi):
            xs.append(float(xi)); ys.append(float(yi)); labels.append(name)
        else:
            print(f"[PHOT][UI] {name}: fuera de FOV (transformación dio NaN).")

    if xs:
        # Pasamos ya en píxeles para evitar heurísticas.
        ui.add_sources(sources={"xcentroid": xs, "ycentroid": ys})
        # (Opcional) Etiquetas pequeñas para distinguirlas visualmente.
        try:
            for xi, yi, name in zip(xs, ys, labels):
                ui.ax.text(xi, yi, name, color="yellow", fontsize=8,
                           ha="left", va="bottom", zorder=11)
            ui.update()
        except Exception:
            pass

def _resolve_png_root_from_ini(ini_path: Optional[Union[str, Path]]) -> Optional[Path]:
    """
    If screening.ini defines [PHOTOMETRY_RESULTS] PNG_FILEPATH,
    return that directory (creating it if needed). Otherwise return None.
    """
    if not ini_path:
        return None

    ini = Path(ini_path).expanduser()
    if not ini.exists():
        return None

    try:
        cfg = ConfigParser()
        cfg.read(ini)
        if cfg.has_section("PHOTOMETRY_RESULTS") and cfg.has_option("PHOTOMETRY_RESULTS", "PNG_FILEPATH"):
            raw = cfg.get("PHOTOMETRY_RESULTS", "PNG_FILEPATH").strip()
            if not raw:
                return None
            root = Path(raw).expanduser()
            root.mkdir(parents=True, exist_ok=True)
            return root
    except Exception as e:
        print(f"[PHOT] WARN: could not read PNG_FILEPATH from INI: {e}")

    return None

_CANONICAL_COLUMNS = [
    "target_name", "fits_file", "observation_id",
    "mag_ab", "mag_err", "count_rate", "count_rate_err",
    "mode", "filter", "zp", "zp_keyword", "zp_source",
    # optional diagnostics if you have them available — leave missing if not:
    "bkg_mean", "bkg_rms", "A_ap_eff", "A_bg_eff",
]

def append_photometry_csv(csv_path: Path, row: dict) -> None:
    """
    Append one row to the CSV at csv_path using canonical columns.
    Creates file+header if missing. Ignores extra keys, fills missing with ''.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # sanitize row to canonical header
    out = {k: row.get(k, "") for k in _CANONICAL_COLUMNS}

    file_exists = csv_path.exists() and csv_path.stat().st_size > 0
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_CANONICAL_COLUMNS)
        if not file_exists:
            w.writeheader()
        w.writerow(out)

# ------------------------------ Helpers --------------------------------------

def _safe_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in (s or ""))

def _png_path_for(
    fits_path: Path,
    target: str,
    ini_path: Optional[Union[str, Path]] = None,
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
    png_root = _resolve_png_root_from_ini(ini_path)
    base_dir = png_root if png_root is not None else fits_path.parent

    # 2) Base name without numeric suffix
    base_name = f"{fits_path.stem}__{_safe_name(target)}"

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

def _find_local_omx_mosaics(
    download_root: Path,
    obsid: str,
    filt: str,
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Look for cached OMX mosaics under:
        <download_root>/<obsid>/<filter>/

    Returns:
        (simag_path, obsmli_path), each may be None if not found.
    """
    obsid = str(obsid)
    filt = str(filt).upper()

    base_dir = download_root / obsid / filt
    if not base_dir.exists():
        return None, None

    simag_path = None
    obsmli_path = None

    # OMX*SIMAG*
    candidates = sorted(base_dir.glob("OMX*SIMAG*"))
    if candidates:
        simag_path = candidates[0]

    # OMX*OBSMLI*
    candidates = sorted(base_dir.glob("OMX*OBSMLI*"))
    if candidates:
        obsmli_path = candidates[0]

    return simag_path, obsmli_path

def _title_for(hduw: HDUW, target: str, obs_id: str, filt: str, pt: PhotTable) -> str:
    fname = Path(getattr(hduw, "file", "")).name or f"OMX{filt}_MOSAIC"
    zp_info = ""
    if getattr(pt, "zero_point", None):
        prov = getattr(pt, "_zp_prov", {}) or {}
        zp_info = f" | ZP={pt.zero_point:.3f} ({prov.get('keyword','?')})"
    return f"{target} | obs={obs_id} | {filt} | {fname}{zp_info}"

def _normalize_selection_to_ap_ann(sel):
    """
    Normalize UI selection to (aperture, annulus-or-None) without using
    truthiness on photutils objects (avoids __len__ TypeError).
    Accepts:
      - dicts with keys like 'aperture', 'annulus', 'box', etc.
      - (aperture, annulus) tuples
      - objects with .rectangular_aperture / .rectangular_annulus
      - bare RectangularAperture / RectangularAnnulus instances
    """
    if sel is None:
        return None, None

    # 1) dict-like
    if isinstance(sel, dict):
        def pick(d, keys):
            for k in keys:
                if k in d:
                    v = d[k]
                    if v is not None:
                        return v
            return None

        ap = pick(sel, ["aperture", "ap", "box", "rectangular_aperture"])
        an = pick(sel, ["annulus", "an", "rectangular_annulus"])
        return ap, an

    # 2) (ap, ann) pair
    if isinstance(sel, (list, tuple)) and len(sel) == 2:
        return sel[0], sel[1]

    # 3) object exposing attributes
    ap = getattr(sel, "rectangular_aperture", None)
    an = getattr(sel, "rectangular_annulus", None)
    if ap is not None or an is not None:
        return ap, an

    # 4) bare photutils apertures
    try:
        from photutils.aperture import RectangularAperture, RectangularAnnulus
        if isinstance(sel, RectangularAperture):
            return sel, None
        if isinstance(sel, RectangularAnnulus):
            return None, sel
    except Exception:
        pass

    # 5) best effort
    return sel, None
    if sel is None:
        return None, None
    if isinstance(sel, dict):
        ap = sel.get("aperture") or sel.get("ap") or sel.get("box") or sel.get("rectangular_aperture")
        an = sel.get("annulus") or sel.get("an") or sel.get("rectangular_annulus")
        return ap, an
    if isinstance(sel, tuple) and len(sel) == 2:
        return sel[0], sel[1]
    return sel, None

def _rate_errors_from_result(res: PhotometryResult):
    c_rate = getattr(res, "count_rate", None)
    A_ap_eff = float(getattr(res, "A_ap_eff", 0) or 0.0)
    A_bg_eff = float(getattr(res, "A_bg_eff", 0) or 0.0)
    bkg_rms = float(getattr(res, "bkg_rms_per_pix", 0) or 0.0)

    if c_rate is None or not np.isfinite(c_rate) or c_rate <= 0:
        return None, None

    var_ap = (bkg_rms ** 2) * A_ap_eff
    var_bg_mean = (bkg_rms ** 2) / max(A_bg_eff, 1.0)
    var_bg_term = (A_ap_eff ** 2) * var_bg_mean
    var_rate = max(var_ap + var_bg_term, 0.0)
    sigma_rate = math.sqrt(var_rate)

    k = 2.5 / math.log(10.0)
    sigma_mag = k * (sigma_rate / c_rate) if c_rate > 0 else None
    return sigma_rate, sigma_mag

def _read_ini_csv_path(ini_path: Optional[Path]) -> Path:
    fallback = Path.home() / "Documents" / "photometry_results.csv"
    if not ini_path:
        return fallback
    try:
        cp = configparser.ConfigParser()
        cp.read(ini_path)
        if "PHOTOMETRY_RESULTS" in cp:
            sec = cp["PHOTOMETRY_RESULTS"]
            for key in ("OUTPUT_CSV", "RESULTS_CSV", "CSV", "PATH", "FILE"):
                if key in sec:
                    p = Path(sec[key]).expanduser()
                    if p.is_dir():
                        return p / "photometry_results.csv"
                    return p
    except Exception:
        pass
    return fallback

def _resolve_download_root_from_ini(ini_path: str | None) -> Path:
    """
    Lee DOWNLOAD_DIRECTORY (case-insensitive) del screening.ini.
    Busca claves: 'DOWNLOAD_DIRECTORY' o 'download_directory' en secciones:
    ['REQUIRED', 'XSA_CRAWLER_HTTP', 'FITS_INTERFACE_DS9'] (por compatibilidad).
    Si no lo encuentra, usa <cwd>/temp y avisa.
    """
    if not ini_path:
        fallback = Path.cwd() / "temp"
        print(f"[PHOT] WARN: INI no especificado; usando fallback {fallback}")
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback

    cfg = ConfigParser()
    cfg.read(ini_path)

    cand_sections = ["REQUIRED", "XSA_CRAWLER_HTTP", "FITS_INTERFACE_DS9"]
    cand_keys = ["DOWNLOAD_DIRECTORY", "download_directory"]

    for sec in cand_sections:
        if cfg.has_section(sec):
            for key in cand_keys:
                if cfg.has_option(sec, key):
                    p = Path(cfg.get(sec, key)).expanduser().resolve()
                    p.mkdir(parents=True, exist_ok=True)
                    return p

    # Si existe sección XSA_CRAWLER_HTTP con 'download_directory' en minúsculas
    # (ya cubierto con el bucle anterior), quedaría capturado.

    fallback = Path.cwd() / "temp"
    print("[PHOT] WARN: No se encontró DOWNLOAD_DIRECTORY en INI; "
          f"usando fallback {fallback}")
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback

def _dest_dir_for(obs_id: str, filt: str, dest_root: Path) -> Path:
    """Construye <dest_root>/<obsid>/<FILTER> y lo crea si no existe."""
    f = (filt or "").strip().upper()
    d = dest_root / str(obs_id) / f
    d.mkdir(parents=True, exist_ok=True)
    return d

def resolve_photometry_csv_from_ini(ini_path):
    """
    Returns (csv_path: Path|None, include_headers: bool).
    Priority:
      1) [PHOTOMETRY_RESULTS] csv_filepath (+ include_headers)
      2) [OUTPUT_RECORDER_CSV] csv_filepath (+ include_headers)  [fallback]
      3) [OBSERVATIONS_REPOSITORY_CSV] csv_filepath (+ include_headers) [fallback]
    """
    if not ini_path:
        print("[PHOT] No INI path passed; will use fallback.")
        return None, True

    ini = Path(ini_path).expanduser()
    if not ini.exists():
        print(f"[PHOT] INI not found: {ini}; will use fallback.")
        return None, True

    cfg = ConfigParser()
    cfg.read(ini)

    def _get(section):
        if cfg.has_section(section) and cfg.has_option(section, "csv_filepath"):
            p = Path(cfg.get(section, "csv_filepath")).expanduser()
            inc = cfg.getboolean(section, "include_headers", fallback=True)
            return p, inc
        return None, None

    # Primary: PHOTOMETRY_RESULTS
    p, inc = _get("PHOTOMETRY_RESULTS")
    if p:
        return p, inc

    # Fallbacks (optional)
    for alt in ("OUTPUT_RECORDER_CSV", "OBSERVATIONS_REPOSITORY_CSV"):
        p, inc = _get(alt)
        if p:
            print(f"[PHOT] Using fallback CSV from [{alt}].")
            return p, inc

    # Nothing found → show what's inside for debugging
    secs = []
    for s in cfg.sections():
        keys = ", ".join(sorted(cfg.options(s)))
        secs.append(f"    [{s}]  {keys}")
    print("[PHOT] No CSV path found in INI. Sections/keys present:\n" + "\n".join(secs))
    return None, True

def zp_from_ini_for_filter(ini_path, filt: str):
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
            f"[PHOT] WARN: Could not parse float ZP from "
            f"[PHOTOMETRY_RESULTS] {key}={raw!r}; falling back to OBSMLI."
        )
        return None, None


def extract_wcs_from_hduw(hduw):
    """
    Try several ways to get a usable celestial WCS from an HDU wrapper.
    Returns a WCS or None.
    """
    # 1) If the wrapper already has a WCS, use it
    w = getattr(hduw, "wcs", None)
    if w is not None:
        try:
            # ensure it's a real celestial WCS
            if hasattr(w, "has_celestial"):
                return w if w.has_celestial else None
            return w
        except Exception:
            pass

    # 2) Try the HDU object inside the wrapper
    h = getattr(hduw, "hdu", None)
    if h is not None and hasattr(h, "header"):
        try:
            w = WCS(h.header)
            if not hasattr(w, "has_celestial") or w.has_celestial:
                return w
        except Exception:
            pass

    # 3) Open the file and search for the first image HDU with a valid WCS
    fpath = str(getattr(hduw, "file", "") or "")
    if fpath:
        try:
            with fits.open(fpath, memmap=False) as hdul:
                # prefer the same index as hduw.hdu if it’s part of this file
                if h is not None and hasattr(h, "name"):
                    try:
                        for hh in hdul:
                            if getattr(hh, "name", None) == h.name and hasattr(hh, "header"):
                                ww = WCS(hh.header)
                                if not hasattr(ww, "has_celestial") or ww.has_celestial:
                                    return ww
                    except Exception:
                        pass

                # otherwise, scan all image extensions
                for hh in hdul:
                    if not hasattr(hh, "data") or hh.data is None:
                        continue
                    try:
                        ww = WCS(hh.header)
                        if not hasattr(ww, "has_celestial") or ww.has_celestial:
                            return ww
                    except Exception:
                        continue

                # last fallback: primary header
                try:
                    ww = WCS(hdul[0].header)
                    if not hasattr(ww, "has_celestial") or ww.has_celestial:
                        return ww
                except Exception:
                    pass
        except Exception:
            pass

    # 4) As a final fallback, try a header attribute on the wrapper
    hdr = getattr(hduw, "header", None)
    if hdr is not None:
        try:
            w = WCS(hdr)
            if not hasattr(w, "has_celestial") or w.has_celestial:
                return w
        except Exception:
            pass

    return None

# --------------------------------- CLI/main ----------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Trail photometry (mosaic-first).")
    p.add_argument("--fits", type=str, required=True)
    p.add_argument("--target-name", type=str, required=True)
    p.add_argument("--observation-id", type=str, required=True)
    p.add_argument("--pos1-ra", type=float, required=True)
    p.add_argument("--pos1-dec", type=float, required=True)
    p.add_argument("--pos2-ra", type=float, required=True)
    p.add_argument("--pos2-dec", type=float, required=True)
    p.add_argument("--filter", type=str, required=True)
    p.add_argument("--ini", type=str, required=False)
    return p.parse_args()

# ---- Build CSV row safely (put this right before _append_csv_row(...)) ----

def _fmt6(x):
    try:
        if x is None:
            return ""
        xf = float(x)
        return "" if not math.isfinite(xf) else f"{xf:.6f}"
    except Exception:
        return ""

# --- CSV appender ------------------------------------------------------------

def _append_csv_row(csv_path, row, include_headers=True):
    """
    Append one photometry result row to CSV, creating file+header if needed.

    Expected columns:
      target_name, fits_file, observation_id, mag_ab, mag_err,
      count_rate, count_rate_err, mode, filter, zp, zp_keyword, zp_source
    """
    cols = [
        "target_name", "fits_file", "observation_id",
        "mag_ab", "mag_err",
        "count_rate", "count_rate_err",
        "mode", "filter",
        "zp", "zp_keyword", "zp_source",
    ]

    p = Path(csv_path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    write_header = not p.exists() or p.stat().st_size == 0

    # Ensure all columns exist; use empty string for missing
    row_out = {k: ("" if row.get(k) is None else row.get(k)) for k in cols}

    with p.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        if write_header:
            writer.writeheader()
        writer.writerow(row_out)

    print(f"[PHOT] CSV appended → {p}")
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    target = args.target_name
    obs_id = str(args.observation_id)
    filt = (args.filter or "").strip().upper()

    # ini_path = Path(args.ini).expanduser() if args.ini else None

    # # Resolve CSV/Download paths from INI first
    # csv_path, include_headers = resolve_photometry_csv_from_ini(getattr(args, "ini", None))
    # if csv_path is None:
    #     # Hard fallback if INI missing/malformed
    #     csv_path = Path.home() / "Documents" / "photometry_results.csv"
    #     print(f"[PHOT] Using fallback CSV: {csv_path}")

    # dest_root = _resolve_download_root_from_ini(getattr(args, "ini", None))
    # dest_dir  = _dest_dir_for(args.observation_id, args.filter, dest_root)

    # # srclist_path, srckind = ensure_om_srclist_or_obsmlist(obs_id, force=False)
    # srclist_path, srckind = ensure_om_srclist_or_obsmlist(
    #     str(obs_id),
    #     force=False,
    #     dest_root=dest_dir,     # <--- ahora se guarda en .../<obsid>/<FILTER>/
    # )

    ini_path = Path(args.ini).expanduser() if args.ini else None

    # Resolve CSV/Download paths from INI first
    csv_path, include_headers = resolve_photometry_csv_from_ini(getattr(args, "ini", None))
    if csv_path is None:
        # Hard fallback if INI missing/malformed
        csv_path = Path.home() / "Documents" / "photometry_results.csv"
        print(f"[PHOT] Using fallback CSV: {csv_path}")

    dest_root = _resolve_download_root_from_ini(getattr(args, "ini", None))
    dest_dir  = _dest_dir_for(args.observation_id, args.filter, dest_root)

    # 1) Try to get ZP directly from INI for this filter
    zp_ini, zp_keyword = zp_from_ini_for_filter(getattr(args, "ini", None), filt)

    # 2) Only if INI ZP is NOT available, resolve/download OBSMLI
    srclist_path = None
    srckind = None
    if zp_ini is None:
        srclist_path, srckind = ensure_om_srclist_or_obsmlist(
            str(obs_id),
            force=False,
            dest_root=dest_dir,   # files go to .../<obsid>/<FILTER>/
        )


    used_mosaic = False
    mosaic_path: Optional[Path] = None
    try:
        # mp = download_om_mosaic_sky_image(obs_id, filt, force=False)
        # 2) Descarga/extrae mosaico y colócalo en el mismo sitio:
        mp = download_om_mosaic_sky_image(
            str(obs_id),
            filter_code=filt,
            force=False,
            dest_root=dest_dir,    # puedes pasar dest_root y dejar que la func construya /<obsid>/<FILTER>
        )
        if mp and Path(mp).exists():
            mosaic_path = Path(mp)
            used_mosaic = True
    except Exception as e:
        print(f"[PHOT] WARN: Mosaic sky image unavailable: {e}")

    chosen_fits = Path(mosaic_path) if used_mosaic else Path(args.fits)
    fits_name = chosen_fits.name

    tn   = getattr(args, "target_name", None) or "unknown"
    obs  = str(getattr(args, "observation_id", "") or "")
    filt = (getattr(args, "filter", "") or "").upper()
    mode = "MOSAIC" if used_mosaic else "CURRENT"

    hduw = HDUW(file=str(chosen_fits))
    arr = np.asarray(
        getattr(hduw, "data", getattr(hduw.hdu, "data", None)),
        dtype=float
    )
    if arr is None or arr.ndim != 2:
        raise ValueError("UI: expected a 2-D image in the HDU.")

    arr = np.ma.masked_invalid(arr)  # what you’re displaying

    # pt = PhotTable(hduw)
    # pt.calibrate_against_source_list(
    #     source_list_file=str(srclist_path),
    #     filter=filt,
    #     source_list_kind=str(srckind),
    # )
    pt = PhotTable(hduw)

    if zp_ini is not None:
        # ZP from screening.ini → no OBSMLI needed
        pt.zero_point = zp_ini
        pt._zp_prov.update(
            keyword=zp_keyword,
            file=str(ini_path) if ini_path is not None else None,
            kind="INI",
        )
        try:
            ini_name = ini_path.name if ini_path is not None else "INI"
        except Exception:
            ini_name = "INI"
        print(
            f"[PHOT] ZP={pt.zero_point:.6f} from {ini_name} "
            f"key={zp_keyword} [INI]"
        )
    elif srclist_path is not None:
        # Fallback: original behaviour using OBSMLI / SRCLIST
        pt.calibrate_against_source_list(
            source_list_file=str(srclist_path),
            filter=filt,
            source_list_kind=str(srckind),
        )
    else:
        print(
            "[PHOT] WARN: No ZP from INI and no SRCLIST file; "
            "zero_point remains unset."
        )


    ui = UI(hduw)
    try:
        title = _title_for(hduw, target=target, obs_id=obs_id, filt=filt, pt=pt)
        ui.update(title=title)
        
    except Exception:
        pass

    pos1 = (float(args.pos1_ra), float(args.pos1_dec))
    pos2 = (float(args.pos2_ra), float(args.pos2_dec))
    # _ui_add_sources_compat(ui, pos1, pos2)
    w = extract_wcs_from_hduw(hduw)
    if w is None:
        print("[PHOT][WCS] WARN: No celestial WCS found; pos1/pos2 will be plotted only if UI accepts raw pixels.")
    # safe_add_sources(ui, w, pos1, pos2)
    # --- Llama a esta función tras crear el UI y obtener 'w' ---
    _plot_pos1_pos2(ui, w, pos1, pos2)
    
    ui.add_sources(pos1, pos2, wcs=w)

    selector = TrailSelector(height=5.0, semi_out=5.0, finalize_on_click=False)

    try:
        sel = ui.select_trail(selector)

    except Exception as e:
        print(f"[PHOT] WARN: UI selection failed: {e}")
        _append_csv_row(csv_path, target=target, obs_id=obs_id, fits_file=fits_name, filt=filt,
                      mode=("MOSAIC" if used_mosaic else "CURRENT"),
                      reason="ui_error")
        return

    ap_box, ann_box = _normalize_selection_to_ap_ann(sel)


    if ap_box is None:
        _append_csv_row(csv_path, target=target, obs_id=obs_id, fits_file=fits_name, filt=filt,
                      mode=("MOSAIC" if used_mosaic else "CURRENT"),
                      reason="no_selection")
        return

    try:
        png_path = _png_path_for(chosen_fits, target, ini_path)
        ui.fig.savefig(png_path, dpi=150, bbox_inches="tight")
        print(f"[PHOT] PNG exported: {png_path}")

    except Exception as e:
        print(f"[PHOT] WARN: PNG export failed: {e}")

    mode_label = "MOSAIC" if used_mosaic else "CURRENT"
    try:
        res: PhotometryResult = pt.perform_trail_photometry(ap_box, ann_box, debug=True)
    except Exception as e:
        print(f"[PHOT] WARN: photometry failed: {e}")
        _append_csv_row(csv_path, target=target, obs_id=obs_id, fits_file=fits_name, filt=filt,
                      mode=mode_label, reason=str(e))
        return

    rate_err = getattr(res, "count_rate_err", None)
    mag_err  = getattr(res, "mag_err", None)
    if rate_err is None or mag_err is None:
        rr, mm = _rate_errors_from_result(res)
        if rate_err is None:
            rate_err = rr
        if mag_err is None:
            mag_err = mm

    print(f"[PHOT] {mode_label}: mag={res.mag_ab}  mag_err={mag_err}  "
          f"rate={res.count_rate}  rate_err={rate_err}  file={fits_name}")

    prov = getattr(pt, "_zp_prov", {}) or {}
    row = {
        "target_name":      tn,
        "fits_file":        fits_name,
        "observation_id":   obs,
        "mag_ab":           _fmt6(getattr(res, "mag_ab", None)),
        "mag_err":          _fmt6(getattr(res, "mag_err", None)),
        "count_rate":       _fmt6(getattr(res, "count_rate", None)),
        "count_rate_err":   _fmt6(getattr(res, "count_rate_err", None)),
        "mode":             mode,
        "filter":           filt,
        "zp":               _fmt6(getattr(res, "zp_ab", None)),
        "zp_keyword":       getattr(res, "zp_keyword", "") or "",
        "zp_source":        Path(getattr(res, "zp_source_file", "") or "").name,
    }

    _append_csv_row(csv_path, row, include_headers=include_headers)


if __name__ == "__main__":
    main()
